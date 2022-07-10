function [U,S,V,X,total_t]=RAUDL(Y,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim,Dinit)
%欠完备字典更新算法
% Input:
%           Y     ------训练样本集矩阵
%           Tdata     ------稀疏度
%           iter     ------迭代次数
%           dict_size     ------原子个数
%           aim_rank     ------目标秩  
%           tol_bcg     ------blockcg1 的收敛阈值
%           itnlim     ------blockcg1 的迭代次数
%     option: 
%             Dinit                   ------初始字典
% Output:
%               D=U*S*V'       ----字典的低秩近似
%               X         ----稀疏编码
%         total_t         ----运算时间
total_tic = tic;
%初始化字典
if nargin < 8 || isempty(Dinit)
   Dinit=init_dict(Y,dict_size);
end
%err=zeros(1,iter);
P=aim_rank+20;  %过采样
%初始的稀疏矩阵
G=Dinit'*Dinit;
DTY=Dinit'*Y;
X = omp(DTY,G,Tdata) ;

%===========================
%主循环：对D低秩近似
for i=1:iter
    Omgc=randn(dict_size,P);   %生成随机矩阵    
    Q = blockcg1(X,Omgc,tol_bcg,itnlim);  
    YXT=Y*X';                             
    Q=YXT*Q;   
    [Q,~]=qr(Q,0);     
    BT = blockcg1(X,YXT'*Q,tol_bcg,itnlim);  
    [Ut,S,Vt]= svd(BT, 'econ');  
    aim_rank=min(aim_rank,size(Ut,2));
    V= Ut(:, 1:aim_rank);
    S =sparse(S(1:aim_rank,1:aim_rank));
    Ut= Q*Vt;
    U= Ut(:, 1:aim_rank);          
    dd=sqrt(colnorms_squared(S*V'));       %D每列的二范数
 if isempty(find(dd==0, 1))==0
          zero_id=find(dd==0);       
          V(zero_id,:)=randn(length(zero_id),size(V,2));
          dd(zero_id)=sqrt(colnorms_squared(S*V(zero_id,:)'));
 end
      V=repmat((1./dd'),[1 size(V,2)]).*V;   %D的每列除以对应列二范数
      
  G=V*(S.^2)*V';  
  DTY=V*S*(U'*Y);
 X = omp(DTY,G,Tdata) ;
   
end

total_t = toc(total_tic);
end


