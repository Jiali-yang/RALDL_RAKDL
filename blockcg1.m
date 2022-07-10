function W = blockcg1(S,H,tol,iter_num)
%Shi Wenya---BCG
if nargin < 3 || isempty(tol)
   tol = 0.01;
end

if nargin < 4 || isempty(iter_num)
   iter_num = 20;
end

[m,n] = size(H);
W = zeros(m,n);
Rb=H;
Pb=Rb;
j=1;
r=1;
% rr=[];
% bb=[];
% pp=[];
h=norm(H,'fro');
while r>tol && j<=iter_num
    %Q=S*Pb;
    %Q=S*(S'*Pb);
    Q=S*(S'*Pb)+0.001*Pb;
    E1A2b=Pb'*Q;
    E=pinv(E1A2b);
    W=W+Pb*E*(Pb'*H);
   
    %b=norm(W,'fro');
    %bb=[bb,b];
    Rb=Rb-Q*E*(Pb'*Rb);
    Pb=Rb-Pb*E*(Q'*Rb);
    %p=norm(Pb,'fro');
    %pp=[pp,p];
    r=norm(H-S*(S'*W),'fro')/h;
    %rr=[rr,r];
  %  W1=orth(W);  %?????
    j=j+1;
end