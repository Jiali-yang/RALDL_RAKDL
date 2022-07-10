function [U_cell,S_cell,V_cell,total_t]=RALDL_train(Train_data,Train_lable,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim)
%�����ֵ�ѧϰ��������ͼ�����ʱ��ѵ������
% Input:
%           Train_data     ------ѵ������������
%           Train_lable     ------ѵ����������ǩ
%           Tdata     ------ϡ���
%           iter     ------��������
%           dict_size     ------ԭ�Ӹ���
%           aim_rank     ------Ŀ����  
%           tol_bcg     ------blockcg1 ��������ֵ
%           itnlim     ------blockcg1 �ĵ�������
% Output:
%               D_i=U_cell{i}*S_cell{i}*V_cell{i}'       ----�ֵ�ĵ��Ƚ���
%         total_t         ----ѵ��ʱ��
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