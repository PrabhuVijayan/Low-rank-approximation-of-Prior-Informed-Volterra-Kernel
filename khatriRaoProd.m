function y = khatriRaoProd(varargin)

    if nargin == 2
        L = varargin{1};
        R = varargin{2};
        [r1,c1] = size(L);
        [r2,c2] = size(R);
        if r1 ~= r2
            error('Matrices should have equal rows - row-wise Kronecker product!');
        else
            y = repmat(L,1,c2).*kron(R,ones(1,c1));
        end
    elseif nargin == 3
        L = varargin{1};
        M = varargin{2};
        R = varargin{3};
        [r1,~] = size(L);
        [r2,~] = size(M);
        [r3,~] = size(R);
        if r1 ~= r2 || r2 ~= r3  ||  r1 ~= r3
            error('Matrices should have equal rows - row-wise Kronecker product!');
        else
            y = khatriRaoProd(L,khatriRaoProd(M,R));
        end
    else
        error('Inputs are more than 3');
    end
    
end