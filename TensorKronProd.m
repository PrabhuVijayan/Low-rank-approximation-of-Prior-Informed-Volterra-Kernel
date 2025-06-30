function C = TensorKronProd(A,B)

    n1 = size(A);
    n2 = size(B);
    d1 = length(n1);
    d2 = length(n2);
    
    % append with ones if the dimensions are not the same (due to Matlab's
    % removal of trailing ones in the dimension)
    if d1 > d2
        n2 = [n2 ones(1,d1-d2)];
    elseif d2 > d1
        n1 = [n1 ones(1,d2-d1)];
    end
    
    c = kron(A(:),B(:)); %compute all entries
    % now reshape the vector into the desired tensor
    permI = [];
    for i = 1:max(d1,d2)
        permI = [permI i i+max(d1,d2)];
    end
    C = reshape(permute(reshape(c,[n2 n1]),permI),n1.*n2);
    
end