function [U,S,V,X,total_t]=RAODL(Y,Tdata,iter,dict_size,t_rank,tol_bcg,itnlim,Dinit)
%过完备字典更新算法
% Input:
%           Y     ------训练样本集矩阵
%           Tdata     ------稀疏度
%           iter     ------迭代次数
%           dict_size     ------原子个数
%           t_rank     ------目标秩  
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
[m,~]=size(Y);    %err=zeros(1,iter);
P=t_rank+20;  %过采样
%初始的稀疏矩阵
G=Dinit'*Dinit;
DTY=Dinit'*Y;
X = omp(DTY,G,Tdata) ;

%===========================
%主循环：对D低秩近似
for i=1:iter
    XYT_Omega=randn(m,P);    %生成随机矩阵 
    XYT=X*Y';            
    XYT_Omega=XYT*XYT_Omega;        
    Q = blockcg1(X,XYT_Omega,tol_bcg,itnlim);  
    [Q,~]=qr(Q,0);
    BT = blockcg1(X,Q,tol_bcg,itnlim);   
    BT=XYT'*BT;
    [Ut,S,Vt]= svd(BT, 'econ');   
    Vt= Q*Vt;
    t_rank=min(t_rank,size(Ut,2));
    U= Ut(:, 1:t_rank);
    S =sparse(S(1:t_rank,1:t_rank));
    V= Vt(:, 1:t_rank);         
    dd=sqrt(colnorms_squared(S*V'));          
%如果dd中有0，V'用一个随机向量替换
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


