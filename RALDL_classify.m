function [accuracy, classify_results, classify_t] = RALDL_classify(num_classes,test,test_lable,U_cell,S_cell,V_cell,Tdata)
%线性字典学习方法用于图像分类时的测试程序
% Input:
%           test     ------测试样本集矩阵
%           test_lable     ------测试样本集标签
%           Tdata     ------稀疏度
%           U_cell,S_cell,V_cell     ------RALDL_train 的输出
%           num_classes     ------类别总数
% Output:
%            accuracy       ----分类精度
%            classify_results       ----分类结果
%            classify_t         ----测试时间

X = cell(num_classes,1);
res = zeros(num_classes,size(test,2));
classify_tic = tic;
h = waitbar(0,'Classifying Test Examples');
for i = 1:num_classes
    G=V_cell{i}*((S_cell{i}).^2)*(V_cell{i})'; 
    DtY=(V_cell{i}*S_cell{i}*(U_cell{i})')*test;
    X{i} = omp(DtY,G,Tdata) ;
    res(i,:) =sqrt(sum((test - U_cell{i}*S_cell{i}*(V_cell{i})'*X{i}).^2)/numel(test));
    waitbar(i/num_classes);
end
close(h);

[~,classify_results] = min(res,[],1);
accuracy = sum(classify_results==test_lable)/(length(test_lable));
classify_t = toc(classify_tic);    
end

