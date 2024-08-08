function [accuracy, classify_results, classify_t] = RALDL_classify(num_classes,test,test_lable,U_cell,S_cell,V_cell,Tdata)
% ========================================================================
% Author: Jiali Yang (yjiali2015@163.com)
% Date: 08-04-2024
% RALDL program for the testing phase of image classification
% Input:
%           test     ------testing examples
%           test_lable     ------test labels
%           Tdata     ------sparsity level
%           U_cell,S_cell,V_cell     ------Output of function RALDL_train
%           num_classes     ------Number of categories
% Output:
%            accuracy       ----Classification accuracy
%            classify_results       ----Classification results
%            classify_t         ----Testing time
% ========================================================================

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

