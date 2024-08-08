function [A_cell,ATKA_cell,total_t]=RAKDL_train(ktrain_params)
% ========================================================================
% Author: Jiali Yang (yjiali2015@163.com)
% Date: 08-04-2024
% RAKDL program for the training phase of image classification
% Output:
%    A_cell--Dictionary matrix obtained from kernel DL algorithm
%    ATKA_cell--A'KA
%    total_t--computation time
% ========================================================================
total_tic = tic;
% Parameters in ktrain_params
Train_data            = ktrain_params.train_images;      % training examples
Train_lable           = ktrain_params.train_labels;      % train labels
num_classes           = ktrain_params.num_classes;       % number of classes in database
dict_size             = ktrain_params.num_atoms;         % number of atoms in each class' dictionary
kernel_choice         = ktrain_params.ker_type;          % kernel type: 'Gaussian','Polynomial'
kervar1               = ktrain_params.ker_param_1;       % main kernel parameter
kervar2               = ktrain_params.ker_param_2;       % secondary kernel parameter
iter                  = ktrain_params.iter;              % number of dictionary learning iterations
Tdata                 = ktrain_params.Tdata;             % cardinality of sparse representations
c_ratio               = ktrain_params.num_sub;           % Number of samples in the kernel matrix approximation
k                     = ktrain_params.aim_rank;          % The target rank of the KA low-rank approximation
itnlim                = ktrain_params.itnlim;            % Maximum number of iterations for BCG to solve the least squares problem
tol_bcg               = ktrain_params.tol_bcg;           % Convergence threshold for BCG
%===========================

train_cell = cell(1,num_classes); ATKA_cell= cell(1,num_classes);
A_cell= cell(1,num_classes); 
%======================================================
for t = 1:num_classes
    train_cell{t} = Train_data(:,Train_lable==t); % divide the training set to different classes
    [A_cell{t},ATKA_cell{t},~]=RAKDL(train_cell{t},Tdata,iter,dict_size,c_ratio,k,kernel_choice, kervar1, kervar2,tol_bcg,itnlim);     
end
total_t = toc(total_tic);
end

