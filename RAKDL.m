function [A,ATKA,X,total_t]=RAKDL(train,Tdata,iter,dict_size,c_ratio,k,kernel_choice, kervar1, kervar2,tol_bcg,itnlim)
%�˷����ֵ�����㷨
% Input:
%   train����ѵ����, 
%   Tdata����ϡ���, 
%   iter������ѭ����������, 
%   dict_size�����ֵ�����,
%   c_ratio�����˾���Nystrom���Ʋ�����, k����KA���Ƚ���Ŀ����, kernel_choice�����˺�������,
%   kervar1, kervar2�����˺����Ĳ���, tol_bcg����blockcg1 ��������ֵ��itnlim����blockcg1 �ĵ�������
% Output:
%    A�������ֵ�
%    ATKA����A'*K*A �Ľ���
%    X����ϡ�����
%    total_t��������ʱ��
total_tic = tic;
ker_params.ker_type=kernel_choice;
ker_params.ker_param_1= kervar1;
ker_params.ker_param_2= kervar2;
P=k+20; %������20
N=size(train,2);
A = zeros(N, dict_size) ;

%===========================
%��һ�����Ժ˾������Nystrom����
permute_vec = randperm(size(train,2));
c=round(c_ratio*size(train,2)); 
supp = permute_vec(1:c);
Y = train(:,supp); % subsample of training
ker_params.X = train;
ker_params.Y = Y;  
C = calc_kernel(train'*Y,ker_params);  %�����������ĺ˾���C
ker_params.X = Y;
W = calc_kernel(Y'*Y,ker_params);  %�����������ĺ˾���W
W=max(W,W');
[V_w,D_w] = eig(W);  
W_pinv=V_w*pinv(D_w)*V_w';  %W���棬���ں˾����Nystrom����K~=C*W_pinv*C'

%===========================
%�ڶ������Ժ˾������� 
 [Q_c,R_c]=qr(C,0);
 [Uw,Sw,~] = svd(R_c*W_pinv*R_c');  %pinv(K)=(Q_c*Uw)*pinv(Sw)*(Q_c*Uw)'

%===========================
%����������ʼ���ֵ����A�����ʼϡ�����X    
randid = randperm(size(train,2));
for j=1:dict_size
    A(randid(j),j) = 1;  %��ʼ�ֵ�
end
ATK=(A'*C*W_pinv)*C';
ATKA=ATK*A;
sigma=diag(ATKA);  %ATKA�Խ�Ԫ���ɵ�������
ATKA=ATKA./repmat(sigma,[1,size(A,2)]); %ATKAÿ�г��Զ�Ӧ�Խ�Ԫ
ATK=ATK./repmat(sigma,[1,size(train,2)]);  %ATKÿ�г���ATKA��Ӧ�Խ�Ԫ
X = omp(ATK,ATKA,Tdata) ;

%===========================
%���Ĳ�����ѭ��
for i=1:iter
    Omgc=randn(size(X,1),P);        %���������˹����
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
%��KA�ĵ��Ƚ�����A
    A=(Q_c*Uw)*(pinv(Sw)*(Q_c*Uw)'*ATK');

%=========================================
%��ATK��A��X
     ATKA=ATK*A;   %�������ǶԳƵ�     
     sigma=diag(ATKA);  %�Խ�Ԫ�γɵ������������ܳ���0
     sigma=sqrt(abs(sigma));
     ATK=ATK./repmat(sigma,[1,size(train,2)]);  %ATKÿ�г���ATKA��Ӧ�����¶Խ�Ԫ 
     A=A./repmat(sigma',[size(train,2),1]);      %Aÿ�г���ATKA��Ӧ�����¶Խ�Ԫ
     sigma=sigma*sigma';
     ATKA=ATKA./sigma; % normalize to norm-1 in feature space
      
     X = omp(ATK,ATKA,Tdata) ;
       
end

total_t = toc(total_tic);
end