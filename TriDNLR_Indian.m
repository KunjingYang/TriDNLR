clear; %clc;
%% ground truth
addpath('Functions','data')
aa  =  load('Noi_Indian.mat');
aa  =  aa.Noi_Indian;
aa  =  aa(1:128,1:128,:);
aaa(:,:,1:98)   = aa(:,:,5:102);
aaa(:,:,99:133) = aa(:,:,113:147);
aaa(:,:,134:182)= aa(:,:,166:214);
cc = double(aaa); cc = max(cc,0);
x1 = max(cc(:));  dd = cc/x1;
S1  = double(dd);  [M, N, L]=size(S1); 

%% Spectral downsampling matrix
spec0 = load( 'AVIRIS_spec.ascii');
spec  = spec0(:,1); % Center wavelength of AVIRIS
wavelength = [450 520; 520 600; 630 690; 760 900; 1550 1750; 2080 2350]; % ETM/Landsat
m_band = size(wavelength,1); % Number of MSI bands
F = zeros(m_band,182);
for b=1:6
    b_i = find(spec>wavelength(b,1),1);
    b_e = find(spec<wavelength(b,2),1,'last');
    F(b,b_i:b_e) = 1/(b_e+1-b_i);
end
T = F(:,1:L);

%% blur and downsamping
S_bar = hyperConvert2D(S1);
sf = 4;   s0 = 1;
BW  =  ones(sf,1)/sf;
BW1 =  psf2otf(BW,[M 1]);
S_w =  ifft(fft(S1).*repmat(BW1,1,N,L)); % blur with the width  mode
 
BH  = ones(sf,1)/sf;
BH1 = psf2otf(BH,[N 1]);
aa  = fft(permute(S_w,[2 1 3])); 
S_h = (aa.*repmat(BH1,1,M,L));
S_h = permute(ifft(S_h),[2 1 3]); % blur with the height mode

Y_h = S_h(s0:sf:end,s0:sf:end,:); % uniform downsamping
Y_h_bar = hyperConvert2D(Y_h); 

manner = 3;            % input 1,2,3,4,5
if     manner == 1;     SNRh = 2;    SNRm = 5;
elseif manner == 2;     SNRh = 10;   SNRm = 15;
elseif manner == 3;     SNRh = 150;   SNRm = 200;
end

%%  simulate LR-HSI and HR_MSI
sigmah  =   sqrt(sum(Y_h_bar(:).^2)/(10^(SNRh/10))/numel(Y_h_bar));
rng(10,'twister') 
Y_h_bar =   Y_h_bar+ sigmah*randn(size(Y_h_bar));
HSI     =   hyperConvert3D(Y_h_bar,M/sf, N/sf );

rng(10,'twister') 
Y       =     T*S_bar; 
sigmam  =     sqrt(sum(Y(:).^2)/(10^(SNRm/10))/numel(Y));
Y       =     Y + sigmam*randn(size(Y));
MSI     =     hyperConvert3D(Y,M,N); 

%%  parameter setting
cg_iter=100;para.tau=1;
if manner == 1
    subspace=3;Rdims=[128,128,3];mu=8e-3;eta=8e-3; para.tao=0.09;
    para.lambda=1e-2;para.r=5; para.r_max=7; para.gamma=1.3;para.gama=200;
elseif manner == 2
    subspace=4;Rdims=[128,128,4];mu=1.8e-2;eta=1e-2; para.tao=0.03;
    para.lambda=0.04;para.r=7; para.r_max=11; para.gamma=1.41;para.gama=180;
elseif manner == 3
    subspace=4;Rdims=[128,128,subspace];mu=1.6e-2;eta=0.009; para.tao=0.011;
    para.lambda=0.03;para.r=15;para.r_max=25; para.gamma=1.37;para.gama=180;
end
%% Initialization
[w,h,~] = size(HSI);          [~,~,s] = size(MSI);  
R1 = Rdims(1);    R2 = Rdims(2);    R3 = Rdims(3);
%  U0,V0,W0
Z1 = reshape(MSI,[M,N*s]);     [U0,~,~] = svds(Z1,R1);    UVW{1} = U0;
D0 = reshape(Z1'*U0,[N,s*R1]); [V0,~,~] = svds(D0,R2);    UVW{2} = V0;  
                                W0= rand(subspace,R3);    UVW{3} = W0;
%  A_hat,B_hat,C_hat
A_hat = rand(R1,para.r,para.r);   ABC{1} = A_hat;
B_hat = rand(para.r,R2,para.r);   ABC{2} = B_hat;
C_hat = rand(para.r,para.r,R3);   ABC{3} = C_hat;   

Mk = zeros(M,N,subspace); 

%%  main
t=clock;
[D,~,~] = svds(Unfold(HSI,size(HSI),3),subspace);   %  Dictionary/subspace
 for k = 1:4
    % subproblem S
    [S,ABC,r] = Sub1_TriBFus_auto(HSI,MSI,Mk,T,BW1,BH1,sf,s0,mu,eta,para,D,ABC,cg_iter);
    para.r=r;
    X = double(ttm(tensor(S),D,3));
    [PSNR] = quality_assessment(double(im2uint8(S1)),double(im2uint8(X)),0,1/sf);
    fprintf('the %d th iteration, the PSNR1 = %0.4f  \n',k,PSNR);
    
    % subproblem G
    if k == 1;    G = S;    end
    G = Sub2_Nonlocal_corr(S,G,mu,para,eta);
    X = double(ttm(tensor(G),D,3));
    [PSNR,RMSE,ERGAS,SAM, ~,SSIM,~,~] = quality_assessment(double(im2uint8(S1)),double(im2uint8(X)),0,1/sf);
    fprintf('the %d th iteration, the PSNR2 = %0.4f  \n',k,PSNR);
    
    % next iteration
    Mk = (mu*G+eta*S)/(mu+eta);
    
    % stopping criterion
    ksi_S = tnorm(grad_q(S,D,HSI,MSI,BW1,BH1,sf,s0,T,1e-4)+mu*(S-G))/(1+tnorm(S)+tnorm(G));
    PK=zeros(M,N,subspace);  L_fix_fft=fft(G,[],3);
    for j = 1:subspace
        [U,B,V] = svd(L_fix_fft(:,:,j),'econ');
        B = diag(B);   B = tidu(B,para.gama);
        r = length(find(B~=0));   B = B(1:r);
        PK(:,:,j) = U(:,1:r)*diag(B)*V(:,1:r)';
    end
    PK = ifft(PK,[],3);
    co = para.tao/mu;
    ksi_G = tnorm(G-prox_tnn(co*PK+S,co))/(1+tnorm(co*PK)+tnorm(S));
    ksi = max(ksi_S,ksi_G);
    fprintf('ksi_S = %0.4f,  ksi_G = %0.4f \n',ksi_S,ksi_G);
    itera = k;
    if ksi< 7.5e-3; break;   end
 end
 total_time = etime(clock,t);
 fprintf('================== Result ====================\n');
 fprintf('PSNR = %0.3f  total_time = %0.1f  itera = %d \n',PSNR,total_time,itera);
 fprintf('SSIM=%0.2f, ERGAS=%0.2f, SAM=%0.2f, RMSE=%0.2f  \n',SSIM,ERGAS,SAM,RMSE);

