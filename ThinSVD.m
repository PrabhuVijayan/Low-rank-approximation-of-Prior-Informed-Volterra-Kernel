function [Q1,S,TNV] = ThinSVD(TN,Ntrain,I,R)
    D = size(TN.core,2);
    % rankR = nchoosek(D+I-1,I-1);
    U1 = reshape(TN.core{1},[Ntrain,I*R(2)]);
    rankR = rank(U1);
    [Q1,S,V] = svd(U1);
    Q1 = Q1(:,1:rankR);
    S = S(1:rankR,1:rankR);
    V = V(:,1:rankR);
    TNV = cell(1,D);
    TNV{1} = reshape(V,[1,rankR,I,R(2)]);
    for d = 2:D
        TNV{d} = TN.core{d};
    end
end