function [face_train,face_test,gnd_train,gnd_test]=LoadFace(DataBase,train_num,group,type)
% Load the data
%     Input:
%          DataBase     -----Select the database to use for the experiment, and the parameters can be 'ORL', 'Yale', 'YaleB' or 'PIE';
%          train_num    -----The number of images each person used for training;
%          group        -----The generated random labels, i.e., the random selection of different faces for training, have a total of 50 groups;
%          type         -----Select the data loading method, parameters include: 'Originar', 'Skelle', 'Nomalize'.
%              'Original'    ----Loads the original image grayscale values
%              'Scale'       ----Map the grayscale value to [0,1].
%              'Normalize'   ----Normalize each face
%     Output:
%          face_train   -----training dataset, where each row represents 1 face;
%          face_test    -----Test a dataset where each row represents 1 face;
%          gnd_train    -----The label of the training dataset, i.e., the category to which each face belongs;
%          gnd_test     -----Test dataset labels, i.e., the categories to which each face belongs.
%     Example:
%          DataBase='ORL';train_num=5;group=1;
%          [face_train,face_test,gnd_train,gnd_test]=loadData(DataBase,train_num,group,'Scale');
%     Written By Yanqi Tan, School of Computer Science and Technology, Soochow University, tyq0502@gmail.com    
%     2011/7/21

%eval(['load ' 'DataBase\' DataBase '_486x640.mat']);    %Load the face database, and there is the dataset FEA and labels GND after loading

%eval(['load ' 'DataBase\' DataBase '_50x40.mat']);    
 
eval(['load ' 'DataBase\' DataBase '_32x32.mat']);    

%eval(['load ' 'DataBase\' DataBase '_64x64.mat']);    

%eval(['load ' 'DataBase\' DataBase '_100x100.mat']);    

%eval(['load ' 'DataBase\' DataBase '_80x80.mat']);    

%eval(['load ' 'DataBase\' DataBase '_92x112.mat']);    

%eval(['load ' 'DataBase\' DataBase '_64x64.mat']);    
 
[nSmp,~] = size(fea);    %nSmp:the number of faces; nFea:The dimensionality of a face, e.g. ORL_32x32:nSmp=400, nFea=32x32=1024
if (~exist('type','var'))
   type='normalize'; % use normalize instead of scale
  %type='scale'; % 
end
 switch lower(type)
     case 'scale'
         maxValue = max(max(fea));      % Divide by the maximum value (the maximum of the entire matrix) and map the pixel value to [0,1].
         fea = fea/maxValue;
     case 'normalize'
         for i=1:nSmp
             fea(i,:) = fea(i,:)./ max(1e-12,norm(fea(i,:)));    % prevents division by 0 and performs vector normalization operations
         end
     case 'original'
     otherwise
         error('Choose the correct way to load your data!');
 end
eval(['load '  'DataBase_Index\',DataBase '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);  
% Load groups to achieve the effect of random sample selection, and there are training sequences trainIdx and test sequences testIdx after loading
% Note: The generation method of the random sequence trainIdx and the test sequence testIdx is shown in the label code in database_Index
face_train = fea(trainIdx,:); 
face_test = fea(testIdx,:);
gnd_train = gnd(trainIdx); 
gnd_test = gnd(testIdx);


% PS: loadData is mainly loaded with two mat files, a face data file, and a label file. It should be noted that:
% type parameter, there are 3 parameters, and the recognition performance will be different in some cases when different data formats are set.
% However, it is clear that the recognition performance under Original and Scale is the same, but it is better to choose Scale, because it is easy to cause problems when the value is too large for matrix multiplication of big data.