function [accuracy, classify_results,  classify_t] = RAKDL_classify(ktrain_params,test,test_lable,dic_cell,ATKA_cell)
% ========================================================================
% Author: Jiali Yang (yjiali2015@163.com)
% Date: 08-04-2024
% RAKDL program for the testing phase of image classification
% Input:
%   ktrain_params.train_images--testing examples
%   ktrain_params.train_labels--test labels
%   ktrain_params.num_classes--Number of categories
%   ktrain_params.ker_type--Kernel function type
%   ktrain_params.ker_param_1--Arguments to the kernel function
%   ktrain_params.ker_param_2--Arguments to the kernel function
%   ktrain_params.Tdata--sparsity level
%   ktrain_params.num_sub--Number of samples in the kernel matrix approximation
%   test--testing examples
%   test_lable--test labels
%   dic_cell--Dictionary matrix obtained from kernel DL algorithm
%   ATKA_cell--A'KA
% Output:
%    accuracy
%    classify_results
%    classify_t---computation time
% ========================================================================


Train_data      = ktrain_params.train_images;      
Train_lable     = ktrain_params.train_labels;      
num_classes     = ktrain_params.num_classes;
ker_type        = ktrain_params.ker_type;
ker_param_1     = ktrain_params.ker_param_1;
ker_param_2     = ktrain_params.ker_param_2;
card            = ktrain_params.Tdata;
c_ratio         =ktrain_params.num_sub;

X = cell(num_classes,1);        
train_cell= cell(num_classes,1);
res = zeros(num_classes,size(test,2));

classify_tic = tic;
h = waitbar(0,'Classifying Test Examples');
for i = 1:num_classes
    train_cell{i} = Train_data(:,Train_lable==i);  
 ker_params.ker_type=ker_type;
 ker_params.ker_param_1= ker_param_1;
 ker_params.ker_param_2= ker_param_2;
 c=round(c_ratio*size(train_cell{i},2)); 
 permute_vec = randperm(size(train_cell{i},2));
 supp = permute_vec(1:c);
 Y = train_cell{i}(:,supp); % subsample of training
 ker_params.X = train_cell{i};
 ker_params.Y = Y;  
 C = calc_kernel(train_cell{i}'*Y,ker_params);  
 ker_params.X = test;
 C_test= calc_kernel(test'*Y,ker_params);
 ker_params.X = Y;
 W = calc_kernel(Y'*Y,ker_params);  
 W=max(W,W');
 [V_w,D_w] = eig(W);  
 W_pinv=V_w*pinv(D_w)*V_w';  
% K_YY=C*W_pinv*C';
   [X{i}, res(i,:)] = myKOMP(dic_cell{i},ATKA_cell{i},card, C,W_pinv,C_test ) ;
    waitbar(i/num_classes);
end
close(h);

[~,classify_results] = min(res,[],1);
accuracy = sum(classify_results==test_lable)/(length(test_lable));
classify_t = toc(classify_tic);    %% classifcation time
end

function [X, Err] = myKOMP(A,ATKA,T0,C,W_pinv,C_test)
ATC=A'*C;
ATKyz=ATC*W_pinv*C_test'; 
X = omp(ATKyz,ATKA,T0) ; 

    B=X'*(ATC*W_pinv*ATC')*X;
    B=diag(B);
    Err = zeros(size(C_test,1),1); %size(K_zy,1)
    for ii=1:size(C_test,1)
            dgz=ones(size(C_test,1),1);
        Err_part= B(ii)- 2*ATKyz(:,ii)'*X(:,ii);
        Err(ii) = sqrt(dgz(ii)+Err_part) ;  
    end
end 