function [U,S,V,X,total_t]=RAODL(Y,Tdata,iter,dict_size,t_rank,tol_bcg,itnlim,Dinit)
%���걸�ֵ�����㷨
% Input:
%           Y     ------ѵ������������
%           Tdata     ------ϡ���
%           iter     ------��������
%           dict_size     ------ԭ�Ӹ���
%           t_rank     ------Ŀ����  
%           tol_bcg     ------blockcg1 ��������ֵ
%           itnlim     ------blockcg1 �ĵ�������
%     option: 
%             Dinit                   ------��ʼ�ֵ�
% Output:
%               D=U*S*V'       ----�ֵ�ĵ��Ƚ���
%               X         ----ϡ�����
%         total_t         ----����ʱ��
total_tic = tic;
%��ʼ���ֵ�
if nargin < 8 || isempty(Dinit)
   Dinit=init_dict(Y,dict_size);  
end
[m,~]=size(Y);    %err=zeros(1,iter);
P=t_rank+20;  %������
%��ʼ��ϡ�����
G=Dinit'*Dinit;
DTY=Dinit'*Y;
X = omp(DTY,G,Tdata) ;

%===========================
%��ѭ������D���Ƚ���
for i=1:iter
    XYT_Omega=randn(m,P);    %����������� 
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
%���dd����0��V'��һ����������滻
     if isempty(find(dd==0, 1))==0
          zero_id=find(dd==0);       
          V(zero_id,:)=randn(length(zero_id),size(V,2));
          dd(zero_id)=sqrt(colnorms_squared(S*V(zero_id,:)'));
      end
      V=repmat((1./dd'),[1 size(V,2)]).*V;   %D��ÿ�г��Զ�Ӧ�ж�����
      
  G=V*(S.^2)*V';  
  DTY=V*S*(U'*Y);
 X = omp(DTY,G,Tdata) ;
end

total_t = toc(total_tic);
end


