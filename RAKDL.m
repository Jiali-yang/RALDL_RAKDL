function [A,ATKA,X,total_t]=RAKDL(train,Tdata,iter,dict_size,c_ratio,k,kernel_choice, kervar1, kervar2,tol_bcg,itnlim)
%核方法字典更新算法
% Input:
%   train——训练集, 
%   Tdata——稀疏度, 
%   iter——主循环迭代次数, 
%   dict_size——字典列数,
%   c_ratio——核矩阵Nystrom近似采样率, k——KA低秩近似目标秩, kernel_choice——核函数类型,
%   kervar1, kervar2——核函数的参数, tol_bcg——blockcg1 的收敛阈值，itnlim——blockcg1 的迭代次数
% Output:
%    A——核字典
%    ATKA——A'*K*A 的近似
%    X——稀疏编码
%    total_t——运算时间
total_tic = tic;
ker_params.ker_type=kernel_choice;
ker_params.ker_param_1= kervar1;
ker_params.ker_param_2= kervar2;
P=k+20; %过采样20
N=size(train,2);
A = zeros(N, dict_size) ;

%===========================
%第一步：对核矩阵进行Nystrom近似
permute_vec = randperm(size(train,2));
c=round(c_ratio*size(train,2)); 
supp = permute_vec(1:c);
Y = train(:,supp); % subsample of training
ker_params.X = train;
ker_params.Y = Y;  
C = calc_kernel(train'*Y,ker_params);  %计算抽出样本的核矩阵C
ker_params.X = Y;
W = calc_kernel(Y'*Y,ker_params);  %计算抽出样本的核矩阵W
W=max(W,W');
[V_w,D_w] = eig(W);  
W_pinv=V_w*pinv(D_w)*V_w';  %W的逆，用于核矩阵的Nystrom近似K~=C*W_pinv*C'

%===========================
%第二步：对核矩阵求逆 
 [Q_c,R_c]=qr(C,0);
 [Uw,Sw,~] = svd(R_c*W_pinv*R_c');  %pinv(K)=(Q_c*Uw)*pinv(Sw)*(Q_c*Uw)'

%===========================
%第三步：初始化字典矩阵A并求初始稀疏矩阵X    
randid = randperm(size(train,2));
for j=1:dict_size
    A(randid(j),j) = 1;  %初始字典
end
ATK=(A'*C*W_pinv)*C';
ATKA=ATK*A;
sigma=diag(ATKA);  %ATKA对角元生成的列向量
ATKA=ATKA./repmat(sigma,[1,size(A,2)]); %ATKA每行除以对应对角元
ATK=ATK./repmat(sigma,[1,size(train,2)]);  %ATK每行除以ATKA对应对角元
X = omp(ATK,ATKA,Tdata) ;

%===========================
%第四步：主循环
for i=1:iter
    Omgc=randn(size(X,1),P);        %生成随机高斯矩阵
    Q = blockcg1(X,Omgc,tol_bcg,itnlim);
    Q=C*W_pinv*((C'*X')*Q);
    [Q,~]=qr(Q,0);     
    ST = blockcg1(X,X*C,tol_bcg,itnlim);
    ST=(Q'*C*W_pinv)*ST';           
    [U1,S1,V1]=svd(ST,'econ');      
    U1= Q*U1;
    U= U1(:, 1:k);
    S =sparse(S1(1:k,1:k));
    V= V1(:, 1:k);                 
    ATK=V*S*U';
    
%===========================
%由KA的低秩近似求A
    A=(Q_c*Uw)*(pinv(Sw)*(Q_c*Uw)'*ATK');

%=========================================
%由ATK和A求X
     ATKA=ATK*A;   %理论上是对称的     
     sigma=diag(ATKA);  %对角元形成的列向量，不能出现0
     sigma=sqrt(abs(sigma));
     ATK=ATK./repmat(sigma,[1,size(train,2)]);  %ATK每行除以ATKA对应根号下对角元 
     A=A./repmat(sigma',[size(train,2),1]);      %A每列除以ATKA对应根号下对角元
     sigma=sigma*sigma';
     ATKA=ATKA./sigma; % normalize to norm-1 in feature space
      
     X = omp(ATK,ATKA,Tdata) ;
       
end

total_t = toc(total_tic);
end