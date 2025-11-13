function [Tur,ABC,P123] =  TriBFus_blind(Y,Z,T,mu,eta,par,Ds,abc,n,P123)
% fprintf(' Sub1_TriBFus is running ... ');
gamma = par.gamma; lambda = par.lambda; tau = par.tau;
mut = (mu+eta)/2;  beta = par.beta; rho = 0.05;
r= par.r;  r1=r; r3=r; 

%% Initialization
[w,h,L] = size(Y); [W,H,s] = size(Z);
L1 = size(Ds,2);
R1 = W;     R2 = H;      R3 = L1;

E1 = eye(W);    E2 = eye(H);    E3 = eye(L1);
P10 = P123{1};  P20 = P123{2};   P30 = P123{3};
%P1=eye(w,W); P2=eye(h,H); P3=eye(s,L);
PDs = P30*Ds;

A_hat = abc{1};       Ah = reshape(A_hat,[r3,R1*r]);
B_hat = abc{2};       Bh = reshape(B_hat,[r1,R2*r3]);
C_hat = abc{3};       Ch = reshape(C_hat,[r,R3*r1]);
%% Main iterate
for iter=1:100
    %fprintf(' Sub1_iteration: %d   ',iter);
    %% Update Ah and U ...
    NN = reshape(permute(reshape(reshape(Ch,[r*R3,r1])*Bh,[r,R3,R2,r3]),[1,4,3,2]),[r*r3,R2*R3]);
    
    FTF1 = reshape(reshape(reshape(reshape(NN,[r*r3*R2,R3])',[R3*r*r3,R2])*(P20')*P20,[R3,r*r3*R2])'*(Ds')*Ds,[r*r3,R2*R3]) * NN';  
    FTF2 = reshape(reshape(reshape(reshape(NN,[r*r3*R2,R3])',[R3*r*r3,R2]),[R3,r*r3*R2])'*(PDs')*PDs,[r*r3,R2*R3]) * NN';
    FTF3 = reshape(reshape(reshape(reshape(NN,[r*r3*R2,R3])',[R3*r*r3,R2]),[R3,r*r3*R2])',[r*r3,R2*R3]) * NN';
    YF1 = (NN * reshape((reshape((reshape(Y,[w*h,L])*Ds)',[R3*w,h])*P20)',[R2*R3,w]))';
    ZF2 = (NN * reshape((reshape((reshape(Z,[W*H,s])*PDs)',[R3*W,H]))',[R2*R3,W]))';
    TF3 = (NN * reshape((reshape((reshape(T,[W*H,L1]))',[R3*W,H]))',[R2*R3,W]))';
     
    Ah = reshape(Ah',[R1,r*r3]);
    H1 = P10'*YF1+tau*ZF2+mut/2*TF3+lambda*Ah; 
    Ak = CG1_sub( FTF1,FTF2,(mut/2)*FTF3,E1,P10,lambda,H1,Ah,n );
    Ak = Ah + gamma*(Ak-Ah); 
    
    H2 = YF1*Ak'+rho*P10;
    P1 = CG1_blind( FTF1,Ak,H2,P10,rho,n );
    P1 = P10+beta*(P1-P10);   
    
    Ak = reshape(Ak,[R1*r,r3])';       Ah = reshape(Ah,[R1*r,r3])';
    
    %% Update Bh and V ...
    NN = reshape(permute(reshape(reshape(Ak,[r3*R1,r])*Ch,[r3,R1,R3,r1]),[1,4,3,2]),[r3*r1,R3*R1]);
    
    FTF1 = reshape(reshape(reshape(reshape(NN,[r3*r1*R3,R1])',[R1*r3*r1,R3])*(Ds')*Ds,[R1,r3*r1*R3])'*(P10)'*P10,[r3*r1,R3*R1]) * NN';  
    FTF2 = reshape(reshape(reshape(reshape(NN,[r3*r1*R3,R1])',[R1*r3*r1,R3])*(PDs')*PDs,[R1,r3*r1*R3])',[r3*r1,R3*R1]) * NN';
    FTF3 = reshape(reshape(reshape(reshape(NN,[r3*r1*R3,R1])',[R1*r3*r1,R3]),[R1,r3*r1*R3])',[r3*r1,R3*R1]) * NN';  
    YF1 = reshape(reshape(reshape(Y,[w*h,L])*Ds,[w,h*R3])'*P10,[h,R3*R1]) * NN';
    ZF2 = reshape(reshape(reshape(Z,[W*H,s])*PDs,[W,H*R3])',[H,R3*R1]) * NN';
    TF3 = reshape(reshape(reshape(T,[W*H,L1]),[W,H*R3])',[H,R3*R1]) * NN';
    
    Bh = reshape(Bh',[R2,r1*r3]);
    H1 = P20'*YF1+tau*ZF2+(mut/2)*TF3+lambda*Bh;            
    Bk = CG1_sub( FTF1,FTF2,(mut/2)*FTF3,E2,P20,lambda,H1,Bh,n );
    Bk = Bh + gamma*(Bk-Bh);  
    
    H2 = YF1*Bk'+rho*P20;
    P2 = CG1_blind( FTF2,Bk,H2,P20,rho,n );
    P2 = P20 +beta*(P2-P20);
    
    Bk = reshape(Bk,[R2*r3,r1])';       Bh = reshape(Bh,[R2*r3,r1])';
    
    %% Update Ch and W ...
    NN = reshape(permute(reshape(reshape(Bk,[r1*R2,r3])*Ak,[r1,R2,R1,r]),[1,4,3,2]),[r1*r,R1*R2]);
    
    FTF1 = reshape(reshape(reshape(reshape(NN,[r1*r*R1,R2])',[R2*r1*r,R1])*(P10)'*P10,[R2,r1*r*R1])'*(P20)'*P20,[r1*r,R1*R2]) * NN'; 
    FTF2 = reshape(reshape(reshape(reshape(NN,[r1*r*R1,R2])',[R2*r1*r,R1]),[R2,r1*r*R1])',[r1*r,R1*R2]) * NN'; 
    YF1 = reshape(reshape(reshape(Y,[w,h*L])'*P1,[h,L*R1])'*P20,[L,R1*R2]) * NN';
    ZF2 = reshape(reshape(reshape(Z,[W,H*s])',[H,s*R1])',[s,R1*R2]) * NN';
    TF3 = reshape(reshape(reshape(T,[W,H*L1])',[H,L1*R1])',[L1,R1*R2]) * NN';
    FTF3 = FTF2;        
    
    Ch = reshape(Ch',[R3,r1*r]);     
    H1 = Ds'*YF1+tau*PDs'*ZF2+(mut/2)*TF3+lambda*Ch;        
    Ck = CG11_sub( FTF1,FTF2,(mut/2)*FTF3,E3,Ds,PDs,lambda,H1,Ch,n ); % 注意W0与W0_hat交换了位置
    Ck = Ch + gamma*(Ck-Ch);
    
    H2 = ZF2*Ck'*Ds'+rho*P30;
    P3 = CG1_blind( FTF3,Ds*Ck,H2,P30,rho,n );
    %P3(P3<0)=0;
    P3 = P30+beta*(P3-P30);  %P3 = P123{3};
    F = P3;
    % for band = 1:size(F,1)
    %     div = sum(F(band,:));
    %     for i = 1:size(F,2)
    %         F(band,i) = F(band,i)/div;
    %     end
    % end
    % P3 = F;
    PDs = P30*Ds;
    
    Ck = reshape(Ck,[R3*r1,r])';  Ch = reshape(Ch,[R3*r1,r])';
   
    %% Next iteration
    relA = norm(Ah-Ak)/norm(Ah);    relB = norm(Bh-Bk)/norm(Bh);    relC = norm(Ch-Ck)/norm(Ch); 
    dX = max([relA,relB,relC]);
    fprintf('rel_change=%0.4f \n',dX);
    if dX < 0.1       
        break;
    end
    Ah = Ak;            Bh = Bk;            Ch = Ck; 
    P10 = P1;           P20 = P2;           P30 = P3;
end

%% Return data
A_hat = reshape(Ak',[R1,r,r3]);
B_hat = reshape(Bk, [r1,R2,r3]);   
C_hat = reshape(reshape(Ck,[r*R3,r1])',[r1,r,R3]);    

Tur = Triple_product(A_hat,B_hat,C_hat);
ABC{1} = A_hat;  ABC{2} = B_hat;  ABC{3} = C_hat;
P123{1}=P1;   P123{2}=P2;    P123{3} = P3;