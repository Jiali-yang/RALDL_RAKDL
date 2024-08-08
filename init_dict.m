function dic=init_dict(train,dict_size)
% Initialize the dictionary matrix
% Input:
%           train     ------Matrix of training sample set
%           dict_size     ------Number of atoms

    data_ids = find(colnorms_squared(train) > 1e-6);   % ensure no zero data elements are chosen
    if ((size(train,2)) >= dict_size) % When the number of columns of the dictionary matrix is less than the sample size
        ind = randperm(length(data_ids));
        ind = data_ids(ind(1:dict_size));
    else   % When the number of columns of the dictionary matrix is greater than the sample size
        ind = randperm(dict_size);
        ind = mod(ind,length(data_ids)) + 1;
        ind = data_ids(ind(1:dict_size));
    end
    dic = train(:,ind) ;
    dic = dic.*repmat(1./sqrt(sum(dic.*dic)),[size(train,1),1]);  % normalize

end