function temp = kronProd(temp,D)
    temp1 = temp;
    for i=2:D
        temp = TensorKronProd(temp,temp1);
    end    
end