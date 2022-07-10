function [face_train,face_test,gnd_train,gnd_test]=LoadFace(DataBase,train_num,group,type)
%��������
%     Input:
%          DataBase     -----ѡ�����ݿ�����ʵ�飬����ѡ��Ϊ'ORL','Yale','YaleB' or 'PIE';
%          train_num    -----ÿ����ѡ������ѵ����ͼƬ����;
%          group        -----���ɵ������ǩ�������ѡȡ��ͬ����������ѵ�����ܹ���50��;
%          type         -----���ݼ��صķ�ʽ������������'Original','Scale','Normalize'.
%              'Original'    ----����ԭʼͼ��Ҷ�ֵ
%              'Scale'       ----���Ҷ�ֵӳ�䵽[0,1]��
%              'Normalize'   ----��ÿ���������й�һ��
%     Output:
%          face_train   -----ѵ�����ݼ�������ÿ�д���1������;
%          face_test    -----�������ݼ�������ÿ�д���1������;
%          gnd_train    -----ѵ�����ݼ���ǩ����ÿ���������ڵ����;
%          gnd_test     -----�������ݼ���ǩ����ÿ���������ڵ����.
%     Example:
%          DataBase='ORL';train_num=5;group=1;
%          [face_train,face_test,gnd_train,gnd_test]=loadData(DataBase,train_num,group,'Scale');
%     Written By ̷���������ݴ�ѧ�������ѧ�뼼��ѧԺ, tyq0502@gmail.com    
%     2011/7/21

%eval(['load ' 'DataBase\' DataBase '_486x640.mat']);    %�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd

%eval(['load ' 'DataBase\' DataBase '_50x40.mat']);    %�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
 
eval(['load ' 'DataBase\' DataBase '_32x32.mat']);    %�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd

%eval(['load ' 'DataBase\' DataBase '_64x64.mat']);    %�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd

%eval(['load ' 'DataBase\' DataBase '_100x100.mat']);    %�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd

%eval(['load ' 'DataBase\' DataBase '_80x80.mat']);    %�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd

%eval(['load ' 'DataBase\' DataBase '_92x112.mat']);    %�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd

%eval(['load ' 'DataBase\' DataBase '_64x64.mat']);    %�����������ݿ�,���غ������ݼ�fea�ͱ�ǩgnd
 
[nSmp,~] = size(fea);    %nSmp:������������ nFea:������ά������ORL_32x32:nSmp=400, nFea=32x32=1024
if (~exist('type','var'))
   type='normalize'; % use normalize instead of scale
  %type='scale'; % 
end
 switch lower(type)
     case 'scale'
         maxValue = max(max(fea));                                 %�������ֵ(������������)��������ֵӳ�䵽[0,1]��
         fea = fea/maxValue;
     case 'normalize'
         for i=1:nSmp
             fea(i,:) = fea(i,:)./ max(1e-12,norm(fea(i,:)));     %��ֹ����0������������һ������
         end
     case 'original'
     otherwise
         error('��ѡȡ��ȷ�����ݼ��ط�ʽ��');
 end
eval(['load '  'DataBase_Index\',DataBase '\' int2str(train_num) 'Train\'  int2str(group) '.mat']);  
%������𣬴ﵽ���ȡ����Ч�������غ�������ѵ��������trainIdx�Ͳ�������testIdx
%ע���������trainIdx�Ͳ�������testIdx�����ɷ�����database_Index�����label����
face_train = fea(trainIdx,:); 
face_test = fea(testIdx,:);
gnd_train = gnd(trainIdx); 
gnd_test = gnd(testIdx);


%PS:  loadData��Ҫ��load����mat�ļ���һ�����������ļ���һ����ǩ�ļ�����Ҫע�����
%     type�������ò�����3�������ò�ͬ�����ݸ�ʽ��ĳЩ����µ�ʶ�����ܻ���ֲ�ͬ
%     ����������ȻOriginal��Scale�µ�ʶ������һ�����������ѡ��Scale����Ϊ���ڴ�
%     ���ݵľ���˷�ʱ��ֵ̫�����׳������⡣