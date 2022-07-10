function [U_cell,S_cell,V_cell,total_t]=RALDL_train(Train_data,Train_lable,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim)
%线性字典学习方法用于图像分类时的训练程序
% Input:
%           Train_data     ------训练样本集矩阵
%           Train_lable     ------训练样本集标签
%           Tdata     ------稀疏度
%           iter     ------迭代次数
%           dict_size     ------原子个数
%           aim_rank     ------目标秩  
%           tol_bcg     ------blockcg1 的收敛阈值
%           itnlim     ------blockcg1 的迭代次数
% Output:
%               D_i=U_cell{i}*S_cell{i}*V_cell{i}'       ----字典的低秩近似
%         total_t         ----训练时间
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