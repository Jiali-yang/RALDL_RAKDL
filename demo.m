clear all;
clc;
addpath('.\packages\OMPbox') % add sparse coding algorithem OMP
load('.\Data\AR_norm2.mat') % Loading the normalized dataset
% Different parameters are used for different datasets
aim_rank_set=0.6;  % Ratio of target rank to dictionary size
Tdata=5;   % Sparsity level
tol_bcg=0.01; % Convergence threshold for blockcg1
itnlim=5;  % Number of iterations for blockcg1
Database='AR'; 
%%Performances on representation error -- RAODL Algorithm
fprintf('\nRAODL Algorithm... ');
dict_size=1200;  % number of atoms
iter=10;
alg='RAODL';
aim_rank=round(aim_rank_set*size(Norm_data,1));
[U1,S1,V1,X1,total_t1]=RAODL(Norm_data,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim);
D=U1*S1*V1'; % Here D is explicitly calculated in order to get the RMSE
RMSE1 = sqrt(sum(reperror2(Norm_data,D,X1))/numel(Norm_data));  
file=strcat(num2str(Database),'_',alg,'.mat');
save(['Results\', file],'U1','S1','V1','X1','RMSE1','total_t1');
fprintf('done!');


%%Performances on representation error -- RAUDL Algorithm
fprintf('\nRAUDL Algorithm... ');
dict_size=960; % number of atoms
alg='RAUDL';
aim_rank=round(aim_rank_set*size(Norm_data,1));
[U1,S1,V1,X1,total_t1]=RAUDL(Norm_data,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim);
D=U1*S1*V1';
RMSE1 = sqrt(sum(reperror2(Norm_data,D,X1))/numel(Norm_data));  
file=strcat(num2str(Database),'_',alg,'.mat');
save(['Results\', file],'U1','S1','V1','X1','RMSE1','total_t1');
fprintf('done!');

%%Performances on representation error -- RAKDL Algorithm
fprintf('\nRAKDL Algorithm... ');
dict_size=1200; % number of atoms
iter=5;
kernel_choice='Gaussian'; 
kervar1=mean(pdist(Norm_data)); 
kervar2=1;  
alg='RAKDL';
c_ratio=0.4;   % Sampling ratio of the approximate kernel matrix
k=0.4*dict_size;  % Target rank for low-rank approximation of KA
[A,~,X,total_t]=RAKDL(Norm_data,Tdata,iter,dict_size,c_ratio,k,kernel_choice, kervar1, kervar2,tol_bcg,itnlim);
K_YY = gram(Norm_data', Norm_data',kernel_choice,kervar2,kervar1) ; 
total_err=trace((eye(size(Norm_data,2))-A*X)'*K_YY*(eye(size(Norm_data,2))-A*X));
RMSE = sqrt(total_err/numel(Norm_data)); 
file=strcat(num2str(Database),'_',alg,'.mat');
save(['Results\', file],'A','X','RMSE','total_t');
fprintf('done!');