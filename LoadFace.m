function [face_train,face_test,gnd_train,gnd_test]=LoadFace(DataBase,train_num,group,type)
%加载数据
%     Input:
%          DataBase     -----选择数据库用于实验，参数选择为'ORL','Yale','YaleB' or 'PIE';
%          train_num    -----每个人选择用于训练的图片张数;
%          group        -----生成的随机标签，即随机选取不同的人脸用于训练，总共有50组;
%          type         -----数据加载的方式，参数包括：'Original','Scale','Normalize'.
%              'Original'    ----加载原始图像灰度值
%              'Scale'       ----将灰度值映射到[0,1]上
%              'Normalize'   ----对每张人脸进行归一化
%     Output:
%          face_train   -----训练数据集，其中每行代表1个人脸;
%          face_test    -----测试数据集，其中每行代表1个人脸;
%          gnd_train    -----训练数据集标签，即每个人脸属于的类别;
%          gnd_test     -----测试数据集标签，即每个人脸属于的类别.
%     Example:
%          DataBase='ORL';train_num=5;group=1;
%          [face_train,face_test,gnd_train,gnd_test]=loadData(DataBase,train_num,group,'Scale');
%     Written By 谭延琪，苏州大学计算机科学与技术学院, tyq0502@gmail.com    
%     2011/7/21

%eval(['load ' 'DataBase\' DataBase '_486x640.mat']);    %加载人脸数据库,加载后有数据集fea和标签gnd

%eval(['load ' 'DataBase\' DataBase '_50x40.mat']);    %加载人脸数据库,加载后有数据集fea和标签gnd
 
eval(['load ' 'DataBase\' DataBase '_32x32.mat']);    %加载人脸数据库,加载后有数据集fea和标签gnd

%eval(['load ' 'DataBase\' DataBase '_64x64.mat']);    %加载人脸数据库,加载后有数据集fea和标签gnd

%eval(['load ' 'DataBase\' DataBase '_100x100.mat']);    %加载人脸数据库,加载后有数据集fea和标签gnd

%eval(['load ' 'DataBase\' DataBase '_80x80.mat']);    %加载人脸数据库,加载后有数据集fea和标签gnd

%eval(['load ' 'DataBase\' DataBase '_92x112.mat']);    %加载人脸数据库,加载后有数据集fea和标签gnd

%eval(['load ' 'DataBase\' DataBase '_64x64.mat']);    %加载人脸数据库,加载后有数据集fea和标签gnd
 
[nSmp,~] = size(fea);    %nSmp:人脸的张数； nFea:人脸的维数，例ORL_32x32:nSmp=400, nFea=32x32=1024
if (~exist('type','var'))
   type='normalize'; % use normalize instead of scale
  %type='scale'; % 
end
 switch lower(type)
     case 'scale'
         maxValue = max(max(fea));                                 %除以最大值(整个矩阵的最大)，将像素值映射到[0,1]上
         fea = fea/maxValue;
     case 'normalize'
         for i=1:nSmp
             fea(i,:) = fea(i,:)./ max(1e-12,norm(fea(i,:)));     %防止除以0，进行向量归一化运算
         end
     case 'original'
     otherwise
         error('请选取正确的数据加载方式！');
 end
eval(['load '  'DataBase_Index\',DataBase '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);  
%加载组别，达到随机取样本效果，加载后有用于训练的序列trainIdx和测试序列testIdx
%注：随机序列trainIdx和测试序列testIdx的生成方法见database_Index里面的label代码
face_train = fea(trainIdx,:); 
face_test = fea(testIdx,:);
gnd_train = gnd(trainIdx); 
gnd_test = gnd(testIdx);


%PS:  loadData主要是load两个mat文件，一个人脸数据文件，一个标签文件。需要注意的是
%     type参数，该参数有3个，设置不同的数据格式在某些情况下的识别性能会出现不同
%     不过，很显然Original和Scale下的识别性能一样，但是最好选择Scale，因为对于大
%     数据的矩阵乘法时数值太大容易出现问题。