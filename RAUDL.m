function [U,S,V,X,total_t]=RAUDL(Y,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim,Dinit)
%Ƿ�걸�ֵ�����㷨
% Input:
%           Y     ------ѵ������������
%           Tdata     ------ϡ���
%           iter     ------��������
%           dict_size     ------ԭ�Ӹ���
%           aim_rank     ------Ŀ����  
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
%err=zeros(1,iter);
P=aim_rank+20;  %������
%��ʼ��ϡ�����
G=Dinit'*Dinit;
DTY=Dinit'*Y;
X = omp(DTY,G,Tdata) ;

%===========================
%��ѭ������D���Ƚ���
for i=1:iter
    Omgc=randn(dict_size,P);   %�����������    
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
    dd=sqrt(colnorms_squared(S*V'));       %Dÿ�еĶ�����
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


