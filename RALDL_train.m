function [U_cell,S_cell,V_cell,total_t]=RALDL_train(Train_data,Train_lable,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim)
% ========================================================================
% Author: Jiali Yang (yjiali2015@163.com)
% Date: 08-04-2024
% RALDL program for the training phase of image classification
% Input:
%           Train_data     ------training examples
%           Train_lable     ------train labels
%           Tdata     ------sparsity level
%           iter     ------number of dictionary learning iterations
%           dict_size     ------number of atoms in each class' dictionary
%           aim_rank     ------target rank  
%           tol_bcg     ------Convergence threshold for BCG
%           itnlim     ------Maximum number of iterations for BCG to solve the least squares problem
% Output:
%               D_i=U_cell{i}*S_cell{i}*V_cell{i}'       ----Low-rank approximation of the dictionary
%         total_t         ----training time
% ========================================================================

total_tic = tic;
num_classes=max(Train_lable);
train_cell = cell(1,num_classes); U_cell= cell(1,num_classes); 
S_cell= cell(1,num_classes);  V_cell= cell(1,num_classes);
for t = 1:num_classes
    train_cell{t} = Train_data(:,Train_lable==t); % divide the training set to different classes
    if dict_size>size(train_cell{t},1)
    [U_cell{t},S_cell{t},V_cell{t}]=RAODL(train_cell{t},Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim);
    else 
    [U_cell{t},S_cell{t},V_cell{t}]=RAUDL(train_cell{t},Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim);   
    end
end
total_t = toc(total_tic);
end