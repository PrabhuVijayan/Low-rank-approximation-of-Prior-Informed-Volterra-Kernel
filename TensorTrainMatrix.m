function [TN,R] = TensorTrainMatrix(Utilde,D)
    [N,I] = size(Utilde);
    TN.core = cell(1,D);
    TN.n = ones(D,4);

    % initialize last core
    TN.n(D,:) = [1,N,I,1];
    TN.core{D} = reshape(Utilde,TN.n(D,:));

    R = ones(D+1,1);
    for d = D:-1:2
        % R(d) = nchoosek(D-d+I,I-1);
        T = reshape(TN.core{d},[N,I*R(d+1)]);
        T = khatriRaoProd(T,Utilde);
        tempT = T;
%         if d == D
%             tempT(:,1) = 0;
%         end
        tempT = reshape(tempT,[N*I,I*R(d+1)]);
        R(d) = rank(tempT);
        [Q,S,V] = svd(tempT,'econ');
        Q = Q(:,1:R(d));
        S = S(1:R(d),1:R(d));
        V = V(:,1:R(d));
        TN.n(d,:) = [R(d),1,I,R(d+1)];
        TN.core{d} = reshape(V',TN.n(d,:));
        TN.n(d-1,:) = [1,N,I,R(d)];
        TN.core{d-1} = reshape(Q*S,TN.n(d-1,:));
    end
end