function [model,y_mod,models,Cost] = vLMnlssWeighted(u,y,model,MaxCount,W,lambda,LambdaJump)
    % Set default value for Levenberg-Marquardt parameter if not specified
    if (nargin < 6) || isempty(lambda)
        lambda = 0; % Later on calculated as the dominant singular value of the Jacobian
    end
    
    % Set default value for LambdaJump
    % Each LambdaJump iterations, the Levenberg-Marquardt parameter is made
    % smaller, so that the algorithm leans more towards a Gauss-Newton
    % algorithm (converges faster) and less to a gradient descent method
    % (converges in a larger range)
    if (nargin < 7) || isempty(LambdaJump)
        LambdaJump = 1001;
    end
    
    % Extract parameters for later use from input arguments
    N                 = length(u); % Number of samples
    [A,C,E,F]         = deal(model.A,model.C,model.E,model.F); %#ok State-space matrices used in analytical calculation Jacobian
    [n,m,p]           = deal(model.n,model.m,model.p); %#ok Number of states, inputs, and outputs used in analytical calculation Jacobian
    [n_nx,n_ny]       = deal(model.n_nx,model.n_ny); % Number of terms in state and output equation
    [xpowers,ypowers] = deal(model.xpowers,model.ypowers); % All possible terms in state and output equation
    xactive           = model.xactive; %#ok Active terms in state equation used in analytical calculation Jacobian
    yactive           = model.yactive; %#ok Active terms in output equation used in analytical calculation Jacobian
    [T1,T2,sat]       = deal(model.T1, model.T2,model.sat); % Transient and saturation parameters
    
    % Compute the (transient-free) modeled output and the corresponding states
    [y_mod,states] = fFilterNLSS(model,u); %#ok: states used in analytical Jacobian calculation script
    
    % Determine if weighting is in frequency or time domain: only implemented for periodic signals.
    NFD = size(W,3); % Number of frequency bins where weighting is specified (e.g. NFD = floor(Npp/2), where Npp is the number of samples in one period and one phase realization for a multisine excitation)
    if isempty(W)
        % No weighting
        FreqWeighting = false; % No frequency weighting
        W = ones(N,p); % No weighting
    elseif NFD > 1
        % Frequency weighting
        FreqWeighting = true; % Frequency weighting
        R = round((N-T2)/NFD/2); % Number of realizations
        if mod(N-T2,R) ~= 0
            error('Transient handling and weighting matrix are incompatible')
        end
    else
        % Time-domain weighting
        FreqWeighting = false; % No frequency weighting
    end
    
    % If T2 is a scalar, it denotes the number of transient points to discard.
    % If it is a vector, it denotes the indices of the points to discard, e.g.
    % when several data sequences were put together.
    without_T2 = fComputeIndicesTransientRemovalArb(T2,N,p); % Samples that are not discarded
    
    % Compute the (weighted) error signal
    err_old = y_mod(without_T2) - y(without_T2); % Error signal (without transient for aperiodic signals)
    if FreqWeighting
        err_old = reshape(err_old,[(N-T2)/R,R,p]);
        err_old = fft(err_old); % Go to the frequency domain
        err_old = err_old(1:NFD,:,:); % Select only the positive half of the spectrum
        err_old = permute(err_old,[3 1 2]); % p x NFD x R
        err_old = fWeightJacobSubSpace(err_old,W,p,NFD,R); % Add weighting
        K_old = fVec(err_old)'*fVec(err_old); % Calculate cost function
        err_old = permute(err_old,[2 3 1]); % NFD x R x p
        err_old = err_old(:); % NFD R p x 1
        err_old = fReIm(err_old); % Split in real and imaginary part
    else
        err_old = err_old.*W(without_T2); % Time-domain weighting
        K_old   = fVec(err_old)'*fVec(err_old); % Compute cost function
    end
    
    % Initialization of Levenberg-Marquardt algorithm
    Count = 1; % Iteration number
    models = []; % Collection of all models after a successful step
    Cost = NaN(MaxCount,1); % Sequence of cost functions after a successful step
    
    % Compute the rms value of the weighted output to later on calculate the
    % relative error after each successful iteration
    if ~FreqWeighting
        rms_y = rms(W(without_T2).*y(without_T2));
    end
    
    % Compute the derivatives of the polynomials zeta and eta
    [xd_powers,xd_coeff] = lfPolyDerivative(xpowers); %#ok Derivatives of zeta used in analytical calculation Jacobian
    [yd_powers,yd_coeff] = lfPolyDerivative(ypowers); %#ok Derivatives of eta used in analytical calculation Jacobian
    
    % Extract the transient part of the input
    uT = u(fComputeIndicesTransient(T1,N),:); % Transient part of the input
    NT = length(uT); % Length of the transient part
    
    % Should the model be stable on the validation set?
    if all(isfield(model,{'u_val','max_out'}))
        stabilisation = true; % Yes, model stable on validation set
    else
        stabilisation = false; % No
    end
    
    % Prepare for Levenberg-Marquardt optimization
    cwd = pwd; % The current working directory
    addpath(cwd); % Add current working directory to the search path
    cd(tempdir); % Switch to the system's temporary folder
    warning('off','MATLAB:pack:InvalidInvocationLocation'); % Disable warning
    pack; % Consolidate workspace memory
    disp('Starting L.M. Optimization...')
    % inittime = clock; % Save current time to estimate the end time during the optimization 
    
    % Levenberg-Marquardt optimization
    while Count < MaxCount
        % Compute the Jacobian w.r.t. the elements in the matrices A, B, E, F, D, and C
        sJacobianAnalytical;
        
        % Add frequency- or time-domain weighting
        if FreqWeighting
            numpar = size(J,2); % Number of parameters
            J = reshape(J,[(N-T2)/R,R,p,numpar]); % Npp x R x p x numpar
            J = fft(J); % Go to the frequency domain
            J = J(1:NFD,:,:,:); % Select only the positive half of the spectrum
            J = permute(J,[3 1 2 4]); % p x NFD x R x numpar
            J = reshape(J,[p NFD R*numpar]); % p x NFD x R numpar
            J = fWeightJacobSubSpace(J,W,p,NFD,R*numpar); % Add weighting
            J = reshape(J,[p NFD R numpar]); % p x NFD x R x numpar
            J = permute(J,[2 3 1 4]); % NFD x R x p x numpar
            J = reshape(J,[NFD*R*p,numpar]); % NFD R p x numpar
            J = fReIm(J); % Split in real and imaginary part
        else
            J = J.*repmat(W(without_T2),1,size(J,2)); % Add time-domain weighting
        end
    
        % (Potentially) improve numerical conditioning
        [J,scaling] = fNormalizeColumns(J); % Normalize columns to unit rms value
    
        K = K_old; % Initial value of the cost function (=> unsuccessful step)
        [U,S,V] = svd(J,0); % Singular value decomposition of the Jacobian
        clear J;
        pack; % Consolidate workspace memory
    
        if isnan(K)
            break % Stop nonlinear optimization (typically when initial model is unstable)
        end
        
        nowexit = false; % nowexit is true when a successfull step was found
        
        % Keep looking for a successful step in the parameter update
        while not(nowexit) && Count <= MaxCount % K >= K_old && Count <= MaxCount
            % Estimate the rank of the Jacobian
            s = diag(S); % Singular values of the Jacobian
            tol = NT*eps(max(s)); % Tolerance to determine the number of rank deficiencies
            r = min(sum(s>tol),length(s)-n^2); % Estimate the rank of the Jacobian (there are at least n^2 rank deficiencies)
            
            % Compute parameter update dtheta = -L*err_old, where L = (J.'*J + lambda*eye)^-1*J.' 
            s = diag(s(1:r)./(s(1:r).^2 + lambda^2*ones(r,1))); % Singular values of L
            dtheta = -V(:,1:r)*s*U(:,1:r).'*err_old; % Parameter update
            dtheta = dtheta./scaling'; % Denormalize the estimates (Jacobian was normalized)
            
            % Update the model and save as model_new
            model_new = lfVec2Par(dtheta,model);
            
            % Compute new model output and states
            [y_mod, states_new] = fFilterNLSS(model_new,u);
            
            % If the model should be stable on the validation set, ...
            if stabilisation
                % ... compute modeled output on validation set
                model_new.T1 = 0; % No transient handling on validation set
                yStab = fFilterNLSS(model_new,model.u_val); % Modeled output
                model_new.T1 = T1; % Transient handling on estimation set
            end
            % If model is not stable on validation set, while this was required, ... 
            if stabilisation && (max(abs(yStab(:))) > model.max_out || any(isnan(yStab(:)))>0)
                K = Inf; % ... set cost function to infinity (=> unsuccessful step), ...
                disp('Unstable for Validation Set');
            else
                % ... else, compute new model error
                err = y_mod(without_T2) - y(without_T2); % Discard transient
                rms_e = rms(err(:)); % Unweighted rms error
                if FreqWeighting
                    err = reshape(err,[(N-T2)/R,R,p]); % Npp x R x p
                    err = fft(err); % Go to the frequency domain
                    err = err(1:NFD,:,:); % Select only the positive half of the spectrum
                    err = permute(err,[3 1 2]); % p x Npp x R
                    err = fWeightJacobSubSpace(err,W,p,NFD,R); % Add weighting
                    K = fVec(err)'*fVec(err); % Weighted least-squares cost function
                    err = permute(err,[2 3 1]); % Npp x R x p
                    err = err(:); % Npp R p x 1
                    err = fReIm(err); % Split in real and imaginary part
                else
                    err = err.*W(without_T2); % Time domain weighting (Multiple Output: vec(err))
                    K = fVec(err)'*fVec(err); % Cost function
                end
            end
            
            % Check successfullness parameter update
            if K >= K_old % Step was not succesful
                if lambda == 0 % lambda was initialed as zero (lambda = 0 should not be possible later on)
                    lambda = S(1,1); % Initial Levenberg-Marquardt parameter = dominant singular value
                else
                    lambda = lambda*sqrt(10); % Lean more towards gradient descent method (converges in larger range)
                end
            elseif isnan(K) % Typically if model is unstable
                lambda = lambda*sqrt(10); % Lean more towards gradient descent method (converges in larger range)
            else % Step was succesful
                lambda = lambda/20; % Lean more towards Gauss-Newton algorithm (converges faster)
                nowexit = true; % Stop searching for successful step
                if FreqWeighting
                    % Display weighted mean square cost function (divided by 2*NFD*R*p) and the condition number of the Jacobian 
                    disp([num2str(Count) ' Cost Function: ' num2str(K/NFD/R/p/2) ' - Cond: ' num2str(S(1,1)/S(r,r))]);
                else
                    % Display rms error normalized with weighted output, the condition number of the Jacobian, the Levenberg Marquardt parameter, and the unweighted rms error
                    disp([num2str(Count) '  RelErr ' num2str(rms_e/rms_y) ' - Cond: ' num2str(S(1,1)/S(r,r)) ...
                        ' - Lambda: ' num2str(lambda) '  RmsErr ' num2str(rms_e)]);
                end
                Cost(Count) = rms_e; % Collect unweighted rms errors after each successful step
            end
            
            % Make Levenberg-Marquardt parameter smaller after LambdaJump iterations
            if (rem(Count,LambdaJump)==0 && Count~=0)
                lambda = lambda/10; % Lean more towards Gauss-Newton method (converges faster)
                disp('Forced jump lambda');
            end
            
            % Update the number of iterations
            Count = Count + 1;
        end
        clear U V;
        
        % Updates after successful step
        if K <= K_old % If step was successful
            K_old = K; % Update cost function to compare with in next iterations
            err_old = err; % Update error to calculate next parameter update
            if nargout >= 3
                try
                    model.u_val = []; % Do not save u_val over and over in all successful models
                catch
                    % Nothing to do
                end
                model.Cost = K; % Save the obtained cost
                models = [models,model]; % Collect models after successful step
            end
            pack; % Consolidate workspace memory
            model = model_new; % u_val saved in final model (best on estimation data), cost not saved
            states = states_new; %#ok States of the model used in analytical calculation of the Jacobian
            A = model.A; %#ok State matrix used in analytical calculation Jacobian
            C = model.C; %#ok Output matrix used in analytical calculation Jacobian
            E = model.E; %#ok Coefficients state nonlinearity used in analytical calculation Jacobian
            F = model.F; %#ok Coefficients output nonlinearity used in analytical calculation Jacobian
            if sat
                satCoeff = model.satCoeff; %#ok obsolete
            end
        end
    end
    y_mod = fFilterNLSS(model,u); % Output of the optimized model
    
    disp('End of L.M. Optimization.')
    cd(cwd); % Return to original working directory
end
    
% ---------------- Help functions ----------------
function [d_powers,d_coeff]=lfPolyDerivative(powers)
%LFPOLYDERIVATIVE Calculate derivative of a multivariate polynomial
d_coeff = permute(powers,[1 3 2]); % Polynomial coefficients of the derivative
n = size(powers,2); % Number of terms
d_powers = repmat(powers,[1 1 n]); % Terms of the derivative
    for i = 1:n
        d_powers(:,i,i) = abs(powers(:,i)-1); % Derivative w.r.t. variable i has one degree less in variable i than original polynomial
                                              % If original polynomial is constant w.r.t. variable i, then the derivative is zero, but
                                              % take abs to avoid a power -1 (zero coefficient anyway)
        %{
        % This would be more correct, but is slower
        d_powers(:,i,i) = powers(:,i) - 1;
        d_powers(powers(:,i) == 0,:,i) = 0;
        %}
    end
end
    
function model = lfVec2Par(delta_vector,model)
    %LFVEC2PAR Update state-space matrices in model from parameter vector update
    % delta_vector is a vector with parameter updates, where the parameters are
    % stored as follows: theta = [vec(A.');
    %                             vec(B.');
    %                             vec(E.');
    %                             vec(F.');
    %                             vec(satCoeff);
    %                             vec(D.');
    %                             vec(C.')]
    A = zeros(size(model.A)); % Initialize state matrix update
    A(:) = delta_vector(1:model.n^2); % Update in state matrix
    model.A = model.A + A'; % Update the state matrix
    delta_vector(1:model.n^2) = []; % Remove update in state matrix from update vector
    
    B = zeros(size(model.B))'; % Initialize input matrix update
    B(:) = delta_vector(1:model.n*model.m); % Update in input matrix
    model.B = model.B + B'; % Update the input matrix
    delta_vector(1:model.n*model.m) = []; % Remove update in input matrix from the update vector
    
    E = zeros(size(model.E))'; % Initialize E matrix update
    E(model.xactive) = delta_vector(1:length(model.xactive)); % Update in E matrix
    model.E = model.E + E'; % Update the E matrix
    delta_vector(1:length(model.xactive)) = []; % Remove update in E matrix from the update vector
    
    F = zeros(size(model.F))'; % Initialize F matrix update
    F(model.yactive) = delta_vector(1:length(model.yactive)); % Update in F matrix
    model.F = model.F + F'; % Update the F matrix
%     model.F = [0.322376158959296	0.599651416381818	0.803867474210604	0.278852678133467	0.747636349799871	0.501124926684889];
    delta_vector(1:length(model.yactive)) = []; % Remove update in F matrix from the update vector
    
    if model.sat
        model.satCoeff = model.satCoeff + delta_vector(1:model.n); % Update saturation coefficients
        delta_vector(1:model.n) = []; % Remove update in saturation coefficients from the update vector
    end
    
    D = zeros(size(model.D))'; % Initialize direct feed-through matrix update
    D(:) = delta_vector(1:model.p*model.m); % Update in direct feed-through matrix
    model.D = model.D + D'; % Update the direct feed-through matrix
%     model.D = 0.309298638493082;
    delta_vector(1:model.p*model.m) = []; % Remove update in direct feed-through matrix from the update vector
    
    C = zeros(size(model.C))'; % Initialize output matrix update
    C(:) = delta_vector(1:model.p*model.n); % Update in output matrix
    model.C = model.C + C'; % Update the output matrix
    delta_vector(1:model.p*model.n) = []; % Remove update in output matrix from the update vector
    
    % Check compatibility of model and update vector
    if ~isempty(delta_vector)
        error('lfVec2Par error: vector is not empty after parsing all parameters')
    end
end