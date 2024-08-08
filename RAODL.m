function [U,S,V,X,total_t]=RAODL(Y,Tdata,iter,dict_size,t_rank,tol_bcg,itnlim,Dinit)
% ========================================================================
% Author: Jiali Yang (yjiali2015@163.com)
% Date: 08-04-2024
% Gang Wu, Jiali Yang, Randomized Algorithms for Large-Scale Dictionary Learning, Submitted to Netural Networks.
% Algorithm 5: A Randomized Algorithm for over-complete and complete Dictionary Learning (m>=n)
% Input:
%           Y     ------training examples
%           Tdata     ------sparsity level
%           iter     ------number of dictionary learning iterations
%           dict_size     ------number of atoms
%           t_rank     ------target rank  
%           tol_bcg     ------Convergence threshold for BCG
%           itnlim     ------Maximum number of iterations for BCG to solve the least squares problem
%     option: 
%             Dinit                   ------The initial dictionary
% Output:
%               D=U*S*V'       ----Low-rank approximation of the dictionary
%               X         ----Sparse representation matrix
%         total_t         ----computation time
% ========================================================================

total_tic = tic;
%Initialize Dictionary
if nargin < 8 || isempty(Dinit)
   Dinit=init_dict(Y,dict_size);  
end
[m,~]=size(Y);    %err=zeros(1,iter);
P=t_rank+20;  
%Initial sparse matrices
G=Dinit'*Dinit;
DTY=Dinit'*Y;
X = omp(DTY,G,Tdata) ;

%===========================
% Main loop: low rank approximation to D
for i=1:iter
    XYT_Omega=randn(m,P);    
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
% If there are zeros in dd, V' is replaced by a random vector
     if isempty(find(dd==0, 1))==0
          zero_id=find(dd==0);       
          V(zero_id,:)=randn(length(zero_id),size(V,2));
          dd(zero_id)=sqrt(colnorms_squared(S*V(zero_id,:)'));
      end
      V=repmat((1./dd'),[1 size(V,2)]).*V;   
      
  G=V*(S.^2)*V';  
  DTY=V*S*(U'*Y);
 X = omp(DTY,G,Tdata) ;
end

total_t = toc(total_tic);
end


