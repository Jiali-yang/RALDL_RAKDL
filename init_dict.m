function dic=init_dict(train,dict_size)
%初始化字典
% Input:
%           train     ------训练样本集矩阵
%           dict_size     ------原子个数

    data_ids = find(colnorms_squared(train) > 1e-6);   % ensure no zero data elements are chosen
    if ((size(train,2)) >= dict_size) %字典列数小于样本量
        ind = randperm(length(data_ids));
        ind = data_ids(ind(1:dict_size));
    else   %字典列数大于样本量
        ind = randperm(dict_size);
        ind = mod(ind,length(data_ids)) + 1;
        ind = data_ids(ind(1:dict_size));
    end
    dic = train(:,ind) ;
    dic = dic.*repmat(1./sqrt(sum(dic.*dic)),[size(train,1),1]);  %归一化

end