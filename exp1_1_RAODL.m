% Performances on representation error
% Sparse representation experiments on normalized datasets to compare the RMSE of RAODL algorithm under different datasets
%% Loading the normalized dataset
%load('AR_norm1.mat')
%load('YaleB_norm1.mat')
%load('USPS_norm1.mat')
%load('COIL100_norm1.mat')
%load('Facescrub_norm1.mat')
%% Different parameters are used for different datasets
aim_rank_set=[0.1,0.2,0.4,0.6,0.8];
Tdata=5;iter=10;tol_bcg=0.01;
itnlim=5;

dict_size=1200;
Database='AR_o';
% Database='YaleB_o';

% dict_size=3000;
% Database='USPS_o';
% Database='COIL100_o';

% dict_size=10000;
% Database='Facescrub_o';
    
    alg='RAODL';
    for j=1:5
        aim_rank=round(aim_rank_set(j)*size(AR_norm1,1));
        [U1,S1,V1,X1,total_t1]=RAODL(AR_norm1,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim);
        D=U1*S1*V1';
        RMSE1 = sqrt(sum(reperror2(AR_norm1,D,X1))/numel(AR_norm1));  
        file=strcat(num2str(Database),alg,'_',num2str(j),'.mat');
        save(['exp1_1Result\', file],'U1','S1','V1','X1','RMSE1','total_t1');
    end