function model = vSaveNLSSmodel(A,B,C,D,E,F,nx,ny,T1,T2)

    % State-space matrices
    model.A = A;
    model.B = B;
    model.C = C;
    model.D = D;
    model.E = E;
    model.F = F;
    
    % State-space matrices linear intialization
    model.lin.A = A;
    model.lin.B = B;
    model.lin.C = C;
    model.lin.D = D;
    
    % Nonlinear degrees
    model.nx = nx; % in state equation
    model.ny = ny; % in output equation
    
    % State-space dimensions + checking their consistency
    [model.n,model.m,model.p] = fSScheckDims(A,B,C,D);
    
    % Nonlinear terms in state equation
    model.xpowers = fCombinations(model.n+model.m,model.nx); % All possible terms
    model.n_nx = size(model.xpowers,1); % Number of terms
%     model.E = zeros(model.n,model.n_nx); % Polynomial coefficients
%     model.xactive = (1:numel(model.E))'; % Active terms
%     
%     % Nonlinear terms in state equation
    model.ypowers = fCombinations(model.n+model.m,ny); % All possible terms
    model.n_ny = size(model.ypowers,1); % Number of terms
%     model.F = zeros(model.p,model.n_ny); % Polynomial coefficients
%     model.yactive = (1:numel(model.F))'; % Active terms
    
    % Transient handling
    model.T1 = T1; % for periodic data
    model.T2 = T2; % for aperiodic data
    
    % Saturating nonlinearity or not (obsolete, sat should be zero)
    if nargin > 10 && sat ~= 0
        warning('Saturating nonlinearity is obsolete, sat should be zero')
    end
    sat = 0;
    model.sat = sat;
    model.satCoeff = ones(model.n,1);

end