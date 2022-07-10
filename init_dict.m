function dic=init_dict(train,dict_size)
%��ʼ���ֵ�
% Input:
%           train     ------ѵ������������
%           dict_size     ------ԭ�Ӹ���

    data_ids = find(colnorms_squared(train) > 1e-6);   % ensure no zero data elements are chosen
    if ((size(train,2)) >= dict_size) %�ֵ�����С��������
        ind = randperm(length(data_ids));
        ind = data_ids(ind(1:dict_size));
    else   %�ֵ���������������
        ind = randperm(dict_size);
        ind = mod(ind,length(data_ids)) + 1;
        ind = data_ids(ind(1:dict_size));
    end
    dic = train(:,ind) ;
    dic = dic.*repmat(1./sqrt(sum(dic.*dic)),[size(train,1),1]);  %��һ��

end