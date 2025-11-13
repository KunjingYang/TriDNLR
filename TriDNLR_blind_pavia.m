clear; %clc;
%% ground truth
addpath('Functions','data')
% S = imread('pavia.tif');   
load('Pavia_Center.mat');
S = pavia;
S = double(S);
S = S(1:256,1:256,11:end); S1 = S/max(S(:)); 
F = load('R.mat');         F = F.R; 
for band = 1:size(F,1)
        div = sum(F(band,:));
        for i = 1:size(F,2)
            F(band,i) = F(band,i)/div;
        end
 end
T = F(:,1:end-11);      [M,N,L] = size(S1);


%% blur and downsamping
sf =4;
sz=[M N];
s0=1;

%% image's psnr
manner = 2;            % input 1,2,3,4,5
if     manner == 1;     SNRh = 2;    SNRm = 5;
elseif manner == 2;     SNRh = 100;   SNRm = 150;
elseif manner == 3;     SNRh = 15;   SNRm = 20;
end


S_bar = hyperConvert2D(S1);        
BW  =  ones(sf,1)/sf;
BW1 =  psf2otf(BW,[M 1]);   
S_w =  ifft(fft(S1).*repmat(BW1,1,N,L)); % blur with the width  mode

BH  = ones(sf,1)/sf;
BH1 = psf2otf(BH,[N 1]);
aa  = fft(permute(S_w,[2 1 3]));
S_h = (aa.*repmat(BH1,1,M,L));
S_h = permute(ifft(S_h),[2 1 3]);       % blur with the height mode

Y_h = S_h(s0:sf:end,s0:sf:end,:);       % uniform downsamping
Y_h_bar = hyperConvert2D(Y_h); 

%%  simulate LR-HSI
sigmah  =   sqrt(sum(Y_h_bar(:).^2)/(10^(SNRh/10))/numel(Y_h_bar));
rng(10,'twister')      
Y_h_bar =   Y_h_bar+ sigmah*randn(size(Y_h_bar)); 
HSI     =   hyperConvert3D(Y_h_bar,M/sf, N/sf );
%%  simulate HR-MSI
rng(10,'twister')              
Y       =     T*S_bar;       
sigmam  =     sqrt(sum(Y(:).^2)/(10^(SNRm/10))/numel(Y));
Y       =     Y + sigmam*randn(size(Y)); 
MSI     =     hyperConvert3D(Y,M,N); 

%%  parameter setting
cg_iter=100;para.tau=1;
if  manner == 1
    subspace=3;Rdims=[256,256,subspace];mu=3e-3;eta=6e-3; para.rho=0.03;
    para.lambda=0.03;para.r=15; para.r_max=10; para.gamma=1.02;para.gama=70;para.tao=0.3;
 elseif manner == 2
     % semiblind
%     subspace=3;Rdims=[256,256,subspace];mu=8e-3;eta=8e-3; para.rho=0.02; para.beta = 0.0;
%     para.lambda=0.03;para.r=13; para.gamma=1.03;para.gama=50;para.tao=0.01;KK=3; 
    
     % blind
    subspace=3;Rdims=[256,256,subspace];mu=5e-3;eta=6e-3; para.rho=0.02; para.beta = 0.01;
    para.lambda=0.03;para.r=20; para.gamma=1.01;para.gama=200;para.tao=0.05; KK=4;
elseif manner == 3
    subspace=2;Rdims=[256,256,subspace];mu=5e-3;eta=6e-3; para.rho=0.03;para.beta = 0.0;
    para.lambda=0.03;para.r=15; para.gamma=1.0;para.gama=120;para.tao=0.05; KK=4; 
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

% P1,P2,P3
P123{1}=eye(w,M); P123{2}=eye(h,N);

MSI_down = MSI(s0:sf:end,s0:sf:end,:);
MSI_d3 = hyperConvert2D(MSI_down);
HSI3 = hyperConvert2D(HSI);
P123{3} = MSI_d3*pinv(HSI3);

Mk = zeros(M,N,subspace); 

%%  main
t=clock;
[D,~,~] = svds(Unfold(HSI,size(HSI),3),subspace);   %  Dictionary/subspace
 for k = 1:KK
    % subproblem S
    [S,ABC,P123] = TriBFus_blind(HSI,MSI,Mk,mu,eta,para,D,ABC,cg_iter,P123);
    X = double(ttm(tensor(S),D,3));
    [PSNR]=quality_assessment(double(im2uint8(S1)),double(im2uint8(X)),0,1/sf);
    fprintf('the %d th iteration, the PSNR1 = %0.4f  \n',k,PSNR);
    
    % subproblem G
    if k == 1;    G = S;    end
    G = Sub2_Nonlocal(S,G,mu,para,eta);
    %G = prox_tnn(S,para.tao/(para.rho+eta));
    X = double(ttm(tensor(G),D,3));
    [PSNR,RMSE,ERGAS,SAM, ~,SSIM,~,~]=quality_assessment(double(im2uint8(S1)),double(im2uint8(X)),0,1/sf);
    fprintf('the %d th iteration, the PSNR2 = %0.4f  \n',k,PSNR);
    
    % next iteration
    Mk = (mu*G+eta*S)/(mu+eta);
    
    % stopping criterion
    ksi_S = tnorm(grad_q(S,D,HSI,MSI,BW1,BH1,sf,s0,T,1e-3)+mu*(S-G))/(1+tnorm(S)+tnorm(G));
    PK=zeros(M,N,subspace);  L_fix_fft=fft(G,[],3);  
    for j = 1:subspace
        [U,B,V] = svd(L_fix_fft(:,:,j),'econ');
        B = diag(B);    B = tidu(B,para.gama);
        r = length(find(B~=0));    B = B(1:r);
        PK(:,:,j) = U(:,1:r)*diag(B)*V(:,1:r)';
    end
    PK = ifft(PK,[],3);
    co = para.tao/mu;
    ksi_G = tnorm(G-prox_tnn(co*PK+S,co))/(1+tnorm(co*PK)+tnorm(S));
    ksi = max(ksi_S,ksi_G);
    fprintf('ksi_S = %0.4f,  ksi_G = %0.4f \n',ksi_S,ksi_G);
    itera = k;
    if ksi< 1e-2; break; end
 end
 total_time = etime(clock,t);
 fprintf('================== Result ====================\n');
 fprintf('PSNR = %0.4f total_time = %0.2f  itera = %d \n',PSNR,total_time,itera);
 fprintf('SSIM=%0.2f, ERGAS=%0.2f, SAM=%0.2f, RMSE=%0.2f  \n',SSIM,ERGAS,SAM,RMSE);
