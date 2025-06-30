% linestyle   = {'-k','-bo','-ko','--r*','--b*','--k*',':rd',':bd',':kd'};
% marker      = {'r*','b*','k*','rd','bd','kd'};

%% Input/Output Data

% Sample size for training and validation data
Ntrain  = numel(uest);
Nval    = numel(uval);

% I/O data and corresponding filter coefficients
Order = max(max(true_model.nx,true_model.ny));

%% Parameters of Volterra Model for estimation

% Number of Inputs and outputs of the system
p = 1; l = 1;

% Order and Memory length - to be optimized to find the best value for hyperparameters
D1 = Order;
M = Memory;
I = p*(M+1)+1;
ker = I^D1;
rankR = nchoosek(D1+I-1,I-1);

% Create matrix U of size N x (D(D-1)+1)M+1
if system == 2
    U = [];
    for i = 1:D1
        U = [U toeplitz(uest(:).^i,[uest(1,1,1)^i zeros(1,M)])];
    end
    % Linear Solver
    [q1,s1,v1] = svd(U,'econ');
    H_kernel = v1*pinv(s1)*q1'*yest(:);
    if vreg == 1
        Regularization
    end
    H = zeros(I*ones(1,D1));
    H(1,2:end,1) = H_kernel(1:M+1);
    for i = 2:I
        H(i,i,1) = H_kernel(I+i-2);
        if D1 > 2
            H(i,i,i) = H_kernel(2*I+i-3);
            if D1 > 3
            H(i,i,i,i) = H_kernel(3*I+i-4);
            end
        end
    end
elseif system == 3 || system == 4
    U = toeplitz(uest(:),[uest(1,1,1) zeros(1,M)]);
    o2 = fCombinations(M+NInputs,2:D1);
    tmp = size(fCombinations(M+NInputs,2),1);
    count = 0; cnt2 = 0;
    for i = 1:size(o2,1)
        count = count + 1;
        [~,col,val] = find(o2(i,:));
        rtmp = range(col);
        if rtmp <= MemoryW
            U(:,M+NInputs+i) = ones(size(uest(:)));
            for j = 1:length(col)
                U(:,M+NInputs+i) = U(:,M+NInputs+i).*U(:,col(j)).^val(j);
            end
        elseif count == M+1-cnt2
            count = 0; cnt2 = cnt2 + 1;
        elseif count == MemoryW+1 && cnt2 == MemoryH
            count = 0;
        end
        if i == tmp
            count = 0; cnt2 = 0; cnt3 = 0;
        end
    end
    U(:,all(U == 0))=[];
    % Linear Solver
    [q1,s1,v1] = svd(U,'econ');
    H_kernel = v1*pinv(s1)*q1'*yest(:);

    Uv = toeplitz(uval(:),[uval(1,1,1) zeros(1,M)]);
    o2 = fCombinations(M+NInputs,2:D1);
    tmp = size(fCombinations(M+NInputs,2),1);
    count = 0; cnt2 = 0;
    for i = 1:size(o2,1)
        count = count + 1;
        [~,col,val] = find(o2(i,:));
        rtmp = range(col);
        if rtmp <= MemoryW
            Uv(:,M+NInputs+i) = ones(size(uval(:)));
            for j = 1:length(col)
                Uv(:,M+NInputs+i) = Uv(:,M+NInputs+i).*Uv(:,col(j)).^val(j);
            end
        elseif count == M+1-cnt2
            count = 0; cnt2 = cnt2 + 1;
        elseif count == MemoryW+1 && cnt2 == MemoryH
            count = 0;
        end
        if i == tmp
            count = 0; cnt2 = 0; cnt3 = 0;
        end
    end
    Uv(:,all(Uv == 0))=[];

elseif system == 1
    if simulation == 0
        Utilde = matrixU(uest(:),1,M,I,ker);
        % Construct Tensor Train matrices
        [TN,Rank] = TensorTrainMatrix(Utilde,D1);
        [Q1,S,TNV] = ThinSVD(TN,Ntrain,I,Rank);
        TNH = cell(size(TNV));
        for d = 1:D1
            if d == 1
                V1r = reshape(TNV{1},I*Rank(2),[]);
                TNH{d} = V1r*inv(S)*Q1'*yest(:);
                TNH{d} = reshape(TNH{d}',[1,l,I,Rank(2)]);
            else
                TNH{d} = TNV{d};
            end
        end
        H_kernel = contractTNH(TNH);
        H = reshape(H_kernel,I*ones(1,D1));

    else
        o2 = fCombinations(M+NInputs,2:D1); 
        U = toeplitz(uest(:),[uest(1,1,1) zeros(1,M)]);
        %U = [U(:,2:end) U(:,1)];
        for i = 1:size(o2,1)
            [~,col,val] = find(o2(i,:));
            U(:,M+NInputs+i) = ones(size(uest(:)));
            for j = 1:length(col)
                U(:,M+NInputs+i) = U(:,M+NInputs+i).*U(:,col(j)).^val(j);
            end
        end
        % Linear Solver
        [q1,s1,v1] = svd(U,'econ');
        H_kernel = v1*pinv(s1)*q1'*yest(:);
        if vreg == 1
            Regularization
        end

        Utilde = matrixU(uest(:),1,M,I,ker);
        % Construct Tensor Train matrices
        [TN,Rank] = TensorTrainMatrix(Utilde,D1);
        [Q1,S,TNV] = ThinSVD(TN,Ntrain,I,Rank);
        TNH = cell(size(TNV));
        for d = 1:D1
            if d == 1
                V1r = reshape(TNV{1},I*Rank(2),[]);
                TNH{d} = V1r*inv(S)*Q1'*yest(:);
                TNH{d} = reshape(TNH{d}',[1,l,I,Rank(2)]);
            else
                TNH{d} = TNV{d};
            end
        end
        H_k = contractTNH(TNH);
        H = reshape(H_k,I*ones(1,D1));
    end
elseif system == 5 || system == 6  || system == 7
    o2 = fCombinations(M+NInputs,2:D1); 
    U = toeplitz(uest(:),[uest(1,1,1) zeros(1,M)]);
    %U = [U(:,2:end) U(:,1)];
    for i = 1:size(o2,1)
        [~,col,val] = find(o2(i,:));
        U(:,M+NInputs+i) = ones(size(uest(:)));
        for j = 1:length(col)
            U(:,M+NInputs+i) = U(:,M+NInputs+i).*U(:,col(j)).^val(j);
        end
    end
    %U = [ones(length(uest(:)),1) U];
    % Linear Solver
    [q1,s1,v1] = svd(U,'econ');
    H_kernel = v1*pinv(s1)*q1'*yest(:);

    Utilde = matrixU(uest(:),1,M,I,ker);
    % Construct Tensor Train matrices
    [TN,Rank] = TensorTrainMatrix(Utilde,D1);
    [Q1,S,TNV] = ThinSVD(TN,Ntrain,I,Rank);
    TNH = cell(size(TNV));
    for d = 1:D1
        if d == 1
            V1r = reshape(TNV{1},I*Rank(2),[]);
            TNH{d} = V1r*inv(S)*Q1'*yest(:);
            TNH{d} = reshape(TNH{d}',[1,l,I,Rank(2)]);
        else
            TNH{d} = TNV{d};
        end
    end
    H_k = contractTNH(TNH);
    H = reshape(H_k,I*ones(1,D1));
end

% Rearrange 2nd Dimension to the end
if system == 2 %|| (system == 1 && simulation == 0)
    r = {[1,3:I,2]};
    for i = 1:Order
        c = repelem({':'},1,Order);
        c(i) = r;
        H = H(c{:});
    end
end

%% Determine Impulse Response

if system == 1 && simulation == 1 || system == 3 || system == 4
    imp = [0; H_kernel(1:M+1)];
elseif ismember(system,5:7)
    imp = H_kernel(1:M+1);
else
    sz = size(H);
    inds = repmat({1},1,ndims(H));
    inds{1} = 1:sz(1);
    imp = zeros(I,1);
    for i = 1:D1
        idx = 1:D1;
        idx(i) = 1;
        idx(1) = i;
        h = permute(H,idx);
        imp = imp + h(inds{:});
    end
    imp(1) = H(1);
end