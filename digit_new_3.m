%��д����ʶ��ʵ��
%���ڸ�˹������ʶ��
Database = 'MNIST';
train_num = 7;
group_set = [1,2,3,4,5,6,7,8,9,10];
type = 'Normalize';
Tdata=5;
iter=5;
dict_size=1000; %Di:784x100 D:784x9000
bate_set=[10,20,30,40];
tol_bcg=0.01;
itnlim=5;
for i=1:10
    group = group_set(i);
    [face_train,face_test,gnd_train,gnd_test]=LoadFace(Database,train_num,group,type);
    train_data = face_train';
    Test_data = face_test';
    train_label = gnd_train';
    test_label = gnd_test';
    [M,N]=size(train_data);
    num_classes=max(train_label);
    sigma=mean(pdist(train_data));
    
     for j=1:4
    test_data=awgn(Test_data,bate_set(j));   
    aim_rank=round(0.4*M);    
        fprintf('\nDictionary and classifier learning by RAODL...')
        alg='RALDL';
        [U_cell,S_cell,V_cell,train_t]=RALDL_train(train_data,train_label,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim);
        [accuracy, ~, classify_t] = RALDL_classify(num_classes,test_data,test_label,U_cell,S_cell,V_cell,Tdata);
        file=strcat(num2str(i),num2str(Database),alg,'_',num2str(j),'.mat');
        save(['exp3_3_1Result\', file],'train_t','accuracy','classify_t');
        fprintf('done!');
       
    %�˷���
    ktrain_params.train_images=train_data;      % training examples
    ktrain_params.train_labels=train_label;      % train labels
    ktrain_params.num_classes=num_classes;       % number of classes in database
    ktrain_params.num_atoms=dict_size;         % number of atoms in each class' dictionary
    ktrain_params.iter=iter;              % number of dictionary learning iterations
    ktrain_params.Tdata=Tdata;             % cardinality of sparse representations
    %�˺�������
    ktrain_params.ker_type='Gaussian';          % kernel type: 'Gaussian','Polynomial'
    ktrain_params.ker_param_1=1;       % main kernel parameter
    ktrain_params.ker_param_2=sigma;       % secondary kernel parameter
%     %RAKDL�Ĳ���
    ktrain_params.itnlim=5;              %BCG����С�������������������
    ktrain_params.tol_bcg=0.01;          %BCG����С���������tol
    
    
        ktrain_params.num_sub=0.2;
        ktrain_params.aim_rank=round(0.4*dict_size);
            
            fprintf('\nDictionary and classifier learning by RAKDL...')
            alg='RAKDL';
            [A_c3,ATKA_c3,train_t]=RAKDL_train(ktrain_params);
            [accuracy, ~,  classify_t] = RAKDL_classify(ktrain_params,test_data,test_label,A_c3,ATKA_c3);
            file=strcat(num2str(i),num2str(Database),alg,'_',num2str(j),'.mat');
            save(['exp3_3_2Result\', file],'train_t','accuracy','classify_t');
            fprintf('done!');


        c=round(0.1*N);
        c_rank=c;
        smp_type='uniform';
        ker_params.ker_type='Gaussian';
        ker_params.ker_param_1=1;
        ker_params.ker_param_2=sigma;
        [train_map, test_map, virtual_train_t, virtual_test_t] = calc_virtual_map(train_data, test_data, ker_params,smp_type,c,c_rank);
           
            a=min(dict_size,size(train_map,1));
            aim_rank=round(0.4*a);
            fprintf('\nDictionary and classifier learning by LKDL+RALDL...')
            alg='LKDL_RALDL';
            [U_cell,S_cell,V_cell,train_t]=RALDL_train(train_map,train_label,Tdata,iter,dict_size,aim_rank,tol_bcg,itnlim);
            [accuracy, ~, classify_t] = RALDL_classify(num_classes,test_map,test_label,U_cell,S_cell,V_cell,Tdata);
            file=strcat(num2str(i),num2str(Database),alg,'_',num2str(j),'.mat');
            save(['exp3_3_3Result\', file],'train_t','accuracy','classify_t','virtual_train_t','virtual_test_t');
            fprintf('done!');     


    end
end

