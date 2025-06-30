function TN_transposed = transposeTN(TN)
    TN_transposed = cell(size(TN));
    for i = 1:size(TN,2)
       TN_transposed{i} = permute(TN{i},[1,3,2,4]);
    end
end