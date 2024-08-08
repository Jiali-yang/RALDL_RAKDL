% Performances on representation error
% Sparse representation experiments on normalized datasets to compare the RMSE of RAKDL algorithm under different datasets
%% Loading the normalized dataset
%load('AR_norm1.mat')
%load('YaleB_norm1.mat')
%load('USPS_norm1.mat')
%load('COIL100_norm1.mat')
%load('Facescrub_norm1.mat')
%% Different parameters are used for different datasets
Tdata=5;iter=5;tol_bcg=0.01;
itnlim=5;

dict_size=1200;
Database='AR_o';
% Database='YaleB_o';

% dict_size=3000;
% Database='USPS_o';
% Database='COIL100_o';

% dict_size=10000;
% Database='Facescrub_o';

kernel_choice='Gaussian'; 
kervar1=mean(pdist(AR_norm1)); 
kervar2=1;  
 
alg='RAKDL';
c_ratio=0.4;
k=0.4*dict_size;
[A,~,X,total_t]=RAKDL(AR_norm1,Tdata,iter,dict_size,c_ratio,k,kernel_choice, kervar1, kervar2,tol_bcg,itnlim);
K_YY = gram(train', train',kernel_choice,kervar2,kervar1) ; 
total_err=trace((eye(size(AR_norm1,2))-A*X)'*K_YY*(eye(size(AR_norm1,2))-A*X));
RMSE = sqrt(total_err/numel(AR_norm1)); 
file=strcat(num2str(Database),alg,'.mat');
save(['exp1_3Result\', file],'A','X','RMSE','total_t');
