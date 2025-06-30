% Estimate PNLSS model
m = 1; % Number of inputs
p = 1; % Number of outputs

% Transient settings
NTrans = N; % Add one period before the start of each realization
T1 = 0; % Number of transient samples and starting indices of each realization
T2 = 0; % No non-periodic transient handling

% Nonlinear terms
nx = true_model.nx; % Nonlinear degrees in state update equation
ny = 2:D1; % Nonlinear degrees in output equation

Memory = size(A,1);
% Choose model order
n = Memory; %min_na;

% Initial model in PNLSS form
model = vSaveNLSSmodel(A,B,C,D,E,F,nx,ny,T1,T2); % Initialize PNLSS model with state-space matrices

% Compute intial model error
% [y_init,states] = fFilterNLSS(model,u); % Output of the initial model on the estimation data
y_init = U*H_kernel;

% modellintest = model; modellintest.T1 = NTrans; % Change transient parameter for linear model on test data
% ytest_lin = fFilterNLSS(modellintest,utest); % Output of the initial linear model on the test data
% err_lin = otest - ytest_lin; % Output error initial linear model on test data
% estlerr = o - y_init; % initial estimation error of nonlinear model

% Levenberg-Marquardt optimization
if vopt == 1
    % Set weighting
    % for kk = size(covY,3):-1:1
    %     W(:,:,kk) = fSqrtInverse(covY(:,:,kk)); % Frequency weighting
    % end
    W = []; % Uncomment for uniform (= no) weighting

    % Set which monomials will be optimized
    model.xactive = fSelectActive(whichtermsx,n,m,n,nx); % Select active monomials in state equation
    model.yactive = fSelectActive(whichtermsy,n,m,p,ny); % Select active monomials in output equation
    
    % Protect for unstable models
    model.uval = uval; % Optionally, a validation input signal can be passed for which the output of the estimated model should remain bounded
    model.max_out = 1000*max(abs(yval)); % Bound on the output of the estimated model (if this bound is violated, the parameter update is considered as unsuccessful)

    [model, y_mod, models] = vLMnlssWeighted(u,y,model,MaxCount,W,lambda); % The actual optimisation of the PNLSS model
    if ~isempty(models), model = models; end
    y_init = y_mod;
end

estnlerr = o - y_init;
if system ~= 4
    rmseerrVolt(iter) = sqrt(immse(o(Memory+1:end),y_init(Memory+1:end)));
    rmseerrVoltn(iter) = sqrt(immse(y(Memory+1:end),y_init(Memory+1:end)));
    rrmseerrVolt(iter) = sqrt(immse(o(Memory+1:end),y_init(Memory+1:end))/sumsqr(o(Memory+1:end)));
    rrmseerrVoltn(iter) = sqrt(immse(y(Memory+1:end),y_init(Memory+1:end))/sumsqr(y(Memory+1:end)));
else
    rmseerrVolt(iter) = sqrt(sum((o(1001:end)-y_init(1001:end)).^2)/99000);
    rrmseerrVolt(iter) = sqrt((sum((o(1001:end)-y_init(1001:end)).^2)/99000)/sumsqr(o(1001:end)));
end

if iter == 1
figure(5)
hold on; set(gca,'TickLabelInterpreter','latex'); 
set(gca,'FontSize',14); plot([o y estnlerr])
% writematrix([o y estnlerr],'dataout.dat','WriteMode','append')
if SimAll == 0
xlabel('Time index','FontSize', 14,'interpreter','latex')
ylabel('Output (errors)','FontSize', 14,'interpreter','latex')
title('Estimation results','FontSize', 14,'interpreter','latex')
legend('True output', sprintf('Noisy output - SNR = %d dB',SNR),...
    sprintf('Volterra-PNLSS error - M = %d, D = %d',Memory,D1),...
    'location','southwest','FontSize', 14,'interpreter','latex')
end
end
disp(['rms(y-y_mod) = ' num2str(rms(o(Memory+1:end)-y_init(Memory+1:end))) ' (= Output error of the best PNLSS model on the estimation data)'])

%% Search best model over the optimisation path on a fresh set of data
valerrs = [];
for i = 1:length(model)
    model(i).T1 = 0; % Only one realization in the validation data
    if vopt == 1
    yval_mod = fFilterNLSS(model(i),uval);  % Output model i on validation data
    else
    yval_mod = Uv*H_kernel;
    end
    valerr = oval(M+1:end) - yval_mod(M+1:end); % Output error model i on validation data
    valerrs = [valerrs; rms(valerr)]; % Collect output errors of all models in a vector
end
[min_valerr,i] = min(valerrs); % Select the best model on the validation data to avoid overfitting
bestmodelVolterra(iter) = model(i);
figure(6)%'Name','Validation results','NumberTitle','off'
hold on; set(gca,'TickLabelInterpreter','latex'); set(gca,'FontSize',14);
if iter == 1
    if vopt == 1, plot(db(valerrs),'-g','LineWidth',2); hold on, end
    plot(i,db(min_valerr),'ks','Markersize',12,"MarkerFaceColor",'k',"MarkerEdgeColor",'k')
    if SimAll == 0
        xlabel('Successful iteration number','FontSize', 14,'interpreter','latex')
        ylabel('Validation error [dB]','FontSize', 14,'interpreter','latex')
        title('Validation results','FontSize', 14,'interpreter','latex')
        if vopt == 1
            legend('Volterra-PNLSS LM optimization',...
                'Best PNLSS model extracted from Volterra kernels',...
                'location','northeast','FontSize', 14,'interpreter','latex')
        else
        legend('PNLSS model extracted from Volterra kernels',...
            'location','northeast','FontSize', 14,'interpreter','latex')
        end
    end
else
    if vopt == 1, plot(db(valerrs),'-g','LineWidth',2,'HandleVisibility','off'); hold on, end
    plot(i,db(min_valerr),'ks','Markersize',12,"MarkerFaceColor",'k',"MarkerEdgeColor",'k','HandleVisibility','off')
end
box on

%% Result on test data

% Compute output error on test data
if vopt == 1
ytest_lin = fFilterNLSS(bestmodelVolterra(iter),utest);
else
if system == 2
    Ut = [];
    for i = 1:D1
        Ut = [Ut toeplitz(utest(:).^i,[utest(1,1,1)^i zeros(1,M)])];
    end
elseif system == 3 || system == 4
    Ut = toeplitz(utest(:),[utest(1,1,1) zeros(1,M)]);
    o2 = fCombinations(M+NInputs,2:D1);
    tmp = size(fCombinations(M+NInputs,2),1);
    count = 0; cnt2 = 0;
    for i = 1:size(o2,1)
        count = count + 1;
        [~,col,val] = find(o2(i,:));
        rtmp = range(col);
        if rtmp <= MemoryW
            Ut(:,M+NInputs+i) = ones(size(utest(:)));
            for j = 1:length(col)
                Ut(:,M+NInputs+i) = Ut(:,M+NInputs+i).*Ut(:,col(j)).^val(j);
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
    Ut(:,all(Ut == 0))=[];
elseif system == 1
    o2 = fCombinations(M+NInputs,2:D1); 
    Ut = toeplitz(utest(:),[utest(1,1,1) zeros(1,M)]);
%     Ut = [Ut(:,2:end) Ut(:,1)];
    for i = 1:size(o2,1)
        [~,col,val] = find(o2(i,:));
        Ut(:,M+NInputs+i) = ones(size(utest(:)));
        for j = 1:length(col)
            Ut(:,M+NInputs+i) = Ut(:,M+NInputs+i).*Ut(:,col(j)).^val(j);
        end
    end
elseif system == 5 || system == 6 || system == 7
    o2 = fCombinations(M+NInputs,2:D1); 
    Ut = toeplitz(utest(:),[utest(1,1,1) zeros(1,M)]);
%     Ut = [Ut(:,2:end) Ut(:,1)];
    for i = 1:size(o2,1)
        [~,col,val] = find(o2(i,:));
        Ut(:,M+NInputs+i) = ones(size(utest(:)));
        for j = 1:length(col)
            Ut(:,M+NInputs+i) = Ut(:,M+NInputs+i).*Ut(:,col(j)).^val(j);
        end
    end
%     Ut = [ones(length(utest(:)),1) Ut];
end
ytest_lin = Ut*H_kernel;
end
valerr = otest - ytest_lin;
if ismember(system,[1:3,5,6,7])
    rmsterrVolt(iter) = sqrt(immse(otest(Memory+1:end),ytest_lin(Memory+1:end)));
    rmsterrVoltn(iter)= sqrt(immse(ytest(Memory+1:end),ytest_lin(Memory+1:end)));
    rrmsterrVolt(iter) = sqrt(immse(otest(Memory+1:end),ytest_lin(Memory+1:end))/sumsqr(otest(Memory+1:end)));
    rrmsterrVoltn(iter)= sqrt(immse(ytest(Memory+1:end),ytest_lin(Memory+1:end))/sumsqr(ytest(Memory+1:end)));
    tmpmean = valerr(Memory+1:end);
    mu_tr(iter) = mean(tmpmean);
    s_tr(iter) = std(tmpmean);
elseif system == 4
    rmsterrVolt(iter) = sqrt(sum((ytest(1001:79000)-ytest_lin(1001:79000)).^2)/78000);
    rrmsterrVolt(iter) = sqrt((sum((ytest(1001:79000)-ytest_lin(1001:79000)).^2)/78000)/sumsqr(ytest(1001:79000)));
    mu_tr(iter) = sum(valerr(1001:end))/87000;
    s_tr(iter) = sqrt(sum((valerr(1001:end)-mu_tr(iter)).^2)/87000);
end

if iter == 1
    figure(8)
    hold on; set(gca,'TickLabelInterpreter','latex'); set(gca,'FontSize',14);
    plot([otest ytest ytest_lin valerr]) % test results: output, linear error and PNLSS error
    if SimAll == 0
        xlabel('Time index')
        ylabel('Output (errors)')
        title('Testing results')
        legend('True output', sprintf('Noisy output - SNR = %d dB',SNR),...
            'Volterra-PNLSS','Volterra-PNLSS error',...
            'location','southwest',...
            'FontSize', 14,'interpreter','latex')
    end
end