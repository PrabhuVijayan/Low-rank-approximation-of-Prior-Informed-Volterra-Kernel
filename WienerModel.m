clear, clc, close all
diary output.txt
rng(0)

simulation  = 0; % 0. use measurement dataset;      1. create simulation dataset
% 1. Wiener;    2. Hammerstein; 3. Wiener-Hammerstein;  
% 4. NSIB;      5. Heat exchanger;  6. CSTR;    7. pH;  
system      = 4; 
if simulation == 1 && ismember(system,4:7)
    error('System not available for simulation.')
end
SimAll      = 0; % 1 - Simulate all 3 methods;      0 - only Volterra
vreg        = 0; % 1 - Regularize Volterra kernel;  0 - Do not regularize (Elastic Net)
vopt        = 0; % 1 - Optimize Volterra solution;  0 - Do not optimize
plot_data   = 1; % 1 - Plot dataset;                0 - Do not plot
inputnoise  = 0; % 1 - Add noise to the input;      0 - Do not add
SNR         = Inf;
if SNR == Inf, Nruns = 1; else, Nruns = 100; end      % Monte-Carlo trial runs

if simulation == 1
    periodic   = 0;     % 1 - Periodic input;              0 - non-periodic
    inputtype  = 2;     % 1 - Multisine;                   2 - Gaussian input
    nlinearity = 2;     % 1 - Polynomial p1,p2,p3 = [1,0.4,0.2]; 2 - Deadzone (-0.3,0.3); 3 - Sigmoid (k=0.8)
    reg        = 0;     % 1 - Regularization using prior;  0 - Do not regularize (Georgios Birpoutsoukis)
    writeout   = 0;     % 1 - Write plot output in a file; 0 - Do not write
    linfilt    = 1;     % 1 - Low pass IIR; 2. High pass IIR; 3. Low pass FIR
    % For WH, allowed combos are: [1,1], [2,2], [3,3]

    dict = containers.Map([1,2,3],{'lowpassiir','highpassiir','lowpassfir'});
    if system == 1 || system == 2
        if linfilt ~= 3
            d = designfilt(dict(linfilt),'DesignMethod','ellip','FilterOrder',3, ...
               'PassbandFrequency',0.5,'PassbandRipple',0.5,'StopbandAttenuation',20);
        else
            d = designfilt(dict(linfilt),'FilterOrder',20,'CutoffFrequency',500,'PassbandRipple',0.5, ...
                    'StopbandAttenuation',20,'SampleRate',2000);
        end

        [b,a] = tf(d);
        coeff = impz(d);
        
        ncoeff = coeff./norm(coeff);
        [Ht,w] = freqz(b,a);
        [Hn,~] = freqz(ncoeff,1,512);
        nb = numel(b)-1; % Order of the numerator
        na = numel(a)-1; % Order of the denominator
        [bn,an] = invfreqz(Hn,w,nb,na);
    else
         if isequal(linfilt,[1,1])
            d(1) = designfilt(dict(linfilt(1)),'DesignMethod','ellip','FilterOrder',3, ...
                   'PassbandFrequency',0.5,'PassbandRipple',0.5,'StopbandAttenuation',20);
            d(2) = designfilt(dict(linfilt(2)),'DesignMethod','ellip','FilterOrder',3, ...
                   'PassbandFrequency',0.4,'PassbandRipple',0.5,'StopbandAttenuation',20);
         elseif isequal(linfilt,[2,2])
            d(1) = designfilt(dict(linfilt(1)),'DesignMethod','ellip','FilterOrder',3, ...
                   'PassbandFrequency',0.5,'PassbandRipple',0.5,'StopbandAttenuation',20);
            d(2) = designfilt(dict(linfilt(2)),'DesignMethod','ellip','FilterOrder',3, ...
                   'PassbandFrequency',0.6,'PassbandRipple',0.5,'StopbandAttenuation',20);
         elseif isequal(linfilt,[3,3])
            d(1) = designfilt('lowpassfir','FilterOrder',20,'CutoffFrequency',600,'PassbandRipple',1, ...
                    'StopbandAttenuation',60,'SampleRate',2000);
            d(2) = designfilt('lowpassfir','FilterOrder',20,'CutoffFrequency',500,'PassbandRipple',1, ...
                    'StopbandAttenuation',60,'SampleRate',2000);
         else
             error('Filter option not available');
         end

         if ~isequal(linfilt,[3,3])
            [b1,a1] = tf(d(1)); [b2,a2] = tf(d(2));
            coeff1 = impz(d(1)); coeff2 = impz(d(2));
         else
            coeff1 = d(1,1).Coefficients(11:end)'; coeff2 = d(1,2).Coefficients(11:end)';
            b1 = coeff1'; b2 = coeff2'; a1 = 1; a2 = 1;
         end
        
        truetf = tf(b1,a1,1)*tf(b2,a2,1);
        ncoeff1 = coeff1./norm(coeff1); ncoeff2 = coeff2./norm(coeff2);
        trueconv = conv(coeff1,coeff2);
        trueconvn = conv(ncoeff1,ncoeff2);
        [H, w] = freqz(trueconv,1,512); % [H, w] = freqz(cell2mat(truetf.Numerator),cell2mat(truetf.Denominator));
        [Hw,~] = freqz(ncoeff1,1,512); 
        [Hh,~] = freqz(ncoeff2,1,512);
        [Hwt,~] = freqz(b1,a1);
        [Hht,~] = freqz(b2,a2);

        nb1 = numel(b1)-1; % Order of the numerator
        na1 = numel(a1)-1; % Order of the denominator
        nb2 = numel(b2)-1; % Order of the numerator
        na2 = numel(a2)-1; % Order of the denominator
        nb = nb1 + nb2;
        na = na1 + na2;

        [bw,aw] = invfreqz(Hw,w,nb1,na1);
        [bh,ah] = invfreqz(Hh,w,nb2,na2);
    end 
end

% Order of linear models and Memory length for Volterra model and curve fit order
if ~ismember(system,3:4) % Wiener and Hammerstein str.
    if simulation == 1
        MemoryV = 25;   % based on the initial impulse response
        cf_ord  = 15;    % Trial and error
        Memoryl = nb;   % based on the order of filter
    else
        MemoryV = 5;  % Trial and error
        cf_ord  = 3;   % Trial and error
        Memoryl = 5;   % Trial and error for BLA
        Memorys = 5;   % Trial and error for SID
    end
else % WH structure
    if simulation == 1
        MemoryW = 20;   % based on the initial impulse response
        MemoryH = 20;   % based on the initial impulse response 
        Memoryl = nb;   % based on the order of filter
        Memorys = nb;   % based on the order of filter
        cf_ord  = 15;   % Trial and error
    else
        MemoryW = 10;   % Trial and error
        MemoryH = 10;   % Trial and error
        Memoryl = 6;   % Trial and error for BLA
        Memorys = 6;   % Trial and error for SID
        cf_ord  = 3;    % Trial and error
    end
    MemoryV = MemoryW+MemoryH;
end

% Settings Levenberg-Marquardt optimization for classical method(s)
if SimAll == 1
    MaxCount    = 50;       % Number of LM optimizations
    lambda      = 1000;     % Starting value LM damping factor
end

clear dict d

%% Automation

if simulation == 0
    % Load dataset
    switch system
        case 1
            load meas202404051123427theSignalWH400_20231012Wiener.mat
            whichtermsx = 'empty';  % Consider 0 monomials in the state update equation
            whichtermsy = 'full';   % Consider all monomials in the output equation
            true_model.nx = 0;
            true_model.ny = 2:3; 
        case 2
            load meas202404051126120theSignalWH400_20231012Hammerstein.mat
            whichtermsx = 'inputsonly';  % Consider only the input monomials in the state update equation
            whichtermsy = 'inputsonly';  % Consider only the input monomials in the output equation
            true_model.nx = 2:3;
            true_model.ny = 2:3;
        case 3
            load meas202404051122132theSignalWH400_20231012WH.mat
            whichtermsx = 'full';  % Consider all monomials in the state update equation
            whichtermsy = 'full';  % Consider all monomials in the output equation
            true_model.nx = 2:3;
            true_model.ny = 2:3;
        case 4
            load WienerHammerBenchMark.mat
            whichtermsx = 'full';  % Consider all monomials in the state update equation
            whichtermsy = 'full';  % Consider all monomials in the output equation
            true_model.nx = 2:3;
            true_model.ny = 2:3;
            u_meas = uBenchMark; y_meas = yBenchMark;
            clear uBenchMark yBenchMark
        case 5
            load exchanger.dat
            u_meas = exchanger(:,2)-mean(exchanger(:,2)); y_meas = exchanger(:,3)-mean(exchanger(:,3));
            Npp = length(exchanger); 
            fs = round(Npp/(exchanger(end,1)-exchanger(1,1))); P = 1; u = u_meas;
            u_meas = reshape(repmat(u,[1 1 P]),[Npp*P,1]);
            y_meas = reshape(repmat(y_meas,[1 1 P]),[Npp*P,1]);
            whichtermsx = 'empty';  % Consider 0 monomials in the state update equation
            whichtermsy = 'full';   % Consider all monomials in the output equation
            true_model.nx = 0;
            true_model.ny = 2:3;
        case 6
            load cstr.dat
            u_meas = cstr(:,2)-mean(cstr(:,2)); y_meas = cstr(:,4)-mean(cstr(:,4)); 
            Npp = length(cstr);
            fs = round(Npp/(cstr(end,1)-cstr(1,1))); P = 1; u = u_meas;
            u_meas = reshape(repmat(u,[1 1 P]),[Npp*P,1]);
            y_meas = reshape(repmat(y_meas,[1 1 P]),[Npp*P,1]);
            whichtermsx = 'empty';  % Consider 0 monomials in the state update equation
            whichtermsy = 'full';   % Consider all monomials in the output equation
            true_model.nx = 0;
            true_model.ny = 2:3;
        case 7
            load pHdata.dat
            u_meas = pHdata(:,3)-mean(pHdata(:,3)); y_meas = pHdata(:,4)-mean(pHdata(:,4));
            Npp = length(pHdata); 
            fs = round(Npp/(pHdata(end,1)-pHdata(1,1))); P = 1; u = u_meas;
            u_meas = reshape(repmat(u,[1 1 P]),[Npp*P,1]);
            y_meas = reshape(repmat(y_meas,[1 1 P]),[Npp*P,1]);
            whichtermsx = 'empty';  % Consider 0 monomials in the state update equation
            whichtermsy = 'full';   % Consider all monomials in the output equation
            true_model.nx = 0;
            true_model.ny = 2:3;
        otherwise
            error('System not available')
    end
else
    fs = 1; 
    Ts = 1/fs;
    switch system
        % change sample size of all system to be same: Npp = 6000
        % Number of samples
        case 1, Npp = 6000; 
        case 2, Npp = 2000;
        case 3, Npp = 2500;   
    end
    R = 1; % Number of phase realizations
    if periodic == 1, P = 6; repp = 1; else, P = 1; repp = 6; end

    % Input signal generation
    if inputtype == 1
        MultiType = 'Odd'; %[3,5,9,13,15,17,21,25,31,33,41,43,47,51,53,57,61,69,79,81,83,95,97,99,101]-1; % no even excited harmonics
        M = round(0.9*Npp/2);   % Last excited line
        RMSu = 0.05;            % Root mean square value for the input signal
        up = [];
        for i = 1:repp
            [u,lines,non_exc_odd,non_exc_even] = fMultisine(Npp, MultiType, M, R); % Multisine signal, excited and detection lines
            u = u/rms(u(:,1))*RMSu; % Scale multisine to the correct rms level
            up = [up; u];
        end
        linestmp = lines;
    else
        up = randn(Npp*repp,R);
    end
    u = up;

    if periodic == 1, u_meas = reshape(repmat(u,[1 1 P]),[Npp*P,1]); else, u_meas = u; end

    switch system
        case 1
            if nlinearity == 1
                y1 = filter(b,a,u);
                y_meas = y1 + 0.4*y1.^2 + 0.2*y1.^3;
            elseif nlinearity == 2
                ll = -0.75; ul = 0.75;
                y_meas = dzone(filter(b,a,u_meas),ul,ll);
            else
                y_meas = 2./(1+exp(-0.8*(filter(b,a,u_meas))))-1;
            end            
            whichtermsx = 'empty';  % Consider 0 monomials in the state update equation
            whichtermsy = 'full';   % Consider all monomials in the output equation
            true_model.nx = 0;
            true_model.ny = 2:3;
        case 2
            if nlinearity == 1
                y_meas = filter(b,a,u + 0.4*u.^2 + 0.2*u.^3);
            elseif nlinearity == 2
                ll = -0.75; ul = 0.75;
                y_meas = filter(b,a,dzone(u_meas,ul,ll));
            else
                y_meas = filter(b,a,2./(1+exp(-0.8*u_meas))-1);
            end            
            whichtermsx = 'inputsonly';  % Consider only the input monomials in the state update equation
            whichtermsy = 'inputsonly';  % Consider only the input monomials in the output equation
            true_model.nx = 2:3;
            true_model.ny = 2:3;
        case 3
            if nlinearity == 1
                y1 = filter(b1,a1,u);
                y2 = y1 + 0.4*y1.^2 + 0.2*y1.^3;
                y_meas = filter(b2,a2,y2);
            elseif nlinearity == 2
                ll = -0.5; ul = 0.5;
                y_meas = filter(b2,a2,dzone(filter(b1,a1,u_meas),ul,ll));
            else
                y_meas = filter(b2,a2,2./(1+exp(-0.8*(filter(b1,a1,u_meas))))-1);
            end
            whichtermsx = 'full';  % Consider all monomials in the state update equation
            whichtermsy = 'full';  % Consider all monomials in the output equation
            true_model.nx = 2:3;
            true_model.ny = 2:3; % To capture even and odd NLs
    end
    clear periodic repp up
end

if system ~= 4, uexc = u; end

%% parameters of system

% SISO Systems
NInputs     = size(u_meas,2);
NOutputs    = size(y_meas,2);

if ~ismember(system,4:7)
    N = Npp;    % Number of samples
    P = 6;      % Number of periods (4 for training, 1 for validation and 1 for performance testing)
    if simulation == 0
        if system == 1 || system == 3
            us = u;
        elseif system == 2
            us = u_meas(1:Npp);
        end
    else
        us = u_meas(1:Npp);
    end
elseif ~ismember(system,5:7)
    Npp = 188000; P = 1; N = length(u_meas); us = u_meas(1:N); uexc = us;
elseif ismember(system,5:6)
    N = Npp; us = u; uexc = us;
else
    N = Npp; us = u; uexc = us;
end
R = 1; % Number of phase realizations
NSamples = length(u_meas);

ts = 1/fs;
f0 = fs/N;                  % Frequency resolution
frequ = 0:f0:fs-f0;         % Frequency vector, Excited frequencies (normalized)
t = 0:ts:(NSamples-1)*ts;

% Input/Output dataset for training, validation and testing
u = reshape(u_meas,[N,R,NInputs,P]); u = permute(u,[1 4 2 3]); 
y = reshape(y_meas,[N,R,NInputs,P]); y = permute(y,[1 4 2 3]); % N x P x R x NInputs

if ~ismember(system,4:7)
    uest = u(:,1:4);            yest = y(:,1:4); 
    uval = u(:,5);              yval = y(:,5);   % 1 period for validation
    utest = u(:,6);             ytest = y(:,6);  % 1 period for testing
elseif system == 4
    uest = u(1:100000);         yest = y(1:100000);
    uval = u(100001:105000);    yval = y(100001:105000);
    utest = u(105001:end);      ytest = y(105001:end);
elseif system == 5
    uest = u(1:3000);           yest = y(1:3000); 
    uval = u(3001:3500);        yval = y(3001:3500);
    utest = u(3501:end);        ytest = y(3501:end);
elseif system == 6
    uest = u(1:5500);           yest = y(1:5500); 
    uval = u(5501:6500);        yval = y(5501:6500);
    utest = u(6501:end);        ytest = y(6501:end);
else
    uest = [u(1:100); u(301:400); u(601:700); u(901:1000); u(1201:1300); u(1501:1600); u(1801:end);];
    uval = [u(101:200); u(401:500); u(701:800); u(1001:1100); u(1301:1400); u(1601:1700)];
    utest = [u(201:300); u(501:600); u(801:900); u(1101:1200); u(1401:1500); u(1701:1800)];

    yest = [y(1:100); y(301:400); y(601:700); y(901:1000); y(1201:1300); y(1501:1600); y(1801:end);];
    yval = [y(101:200); y(401:500); y(701:800); y(1001:1100); y(1301:1400); y(1601:1700)];
    ytest = [y(201:300); y(501:600); y(801:900); y(1101:1200); y(1401:1500); y(1701:1800)];
end

% Degree of nonlinearity in state and output equations
true_model.m = NInputs;
true_model.p = NOutputs;

clear f0 fs NSamples s_meas signalFilename

%% Plot dataset

if plot_data == 1
    U = fft(us); Ym = fft(y_meas);
%     U = fft(u); Ym = fft(y_meas);
%     U = fft(uest); Ym = fft(yest);
    pos = find(U==0); U(pos) = rand(size(pos))*1e-16;
    pos = find(Ym==0); Ym(pos) = rand(size(pos))*1e-16;
    excind = find(abs(U) > std(uexc));
    excP = (excind - 1)*P + 1;
    figure('Name','Training, Validation and Test dataset','NumberTitle','off'),
    set(gca,'TickLabelInterpreter','latex'); set(gca,'FontSize',14);
    subplot(2,2,1); plot(t(1:length(uest(:))),uest(:),'b', ...
        t(length(uest(:))+1:length(uest(:))+length(uval(:))),uval(:),'r', ...
        t(length(uest(:))+length(uval(:))+1:end),utest(:),'g'); 
    xlim([0,t(end)])
    title('Time domain measurements','FontSize', 14,'interpreter','latex')
    xlabel("time (s)",'FontSize', 14,'interpreter','latex'); 
    ylabel("Input amplitude",'FontSize', 14,'interpreter','latex');
    legend(["Estimation","Validation","Testing"],'Orientation','horizontal','FontSize', 14,'interpreter','latex')
    subplot(2,2,3); plot(t(1:length(uest(:))),yest(:),'b', ...
        t(length(uest(:))+1:length(uest(:))+length(uval(:))),yval(:),'r', ...
        t(length(uest(:))+length(uval(:))+1:end),ytest(:),'g');
    xlim([0,t(end)])
    xlabel("time (s)",'FontSize', 14,'interpreter','latex');
    ylabel("Output amplitude",'FontSize', 14,'interpreter','latex');
    legend(["Estimation","Validation","Testing"],'Orientation','horizontal','FontSize', 14,'interpreter','latex')

if simulation == 0
    lines = excind(1:ceil(length(excind)/2));
    non_exc_odd = setdiff(2:2:length(U),excind);
    non_exc_even = setdiff(1:2:length(U),excind);
    noddP = (non_exc_odd - 1)*P + 1;
    nevenP = (non_exc_even - 1)*P + 1;
    subplot(2,2,2); hold on;
    plot(lines,db(U(excind(1:ceil(length(excind)/2)))),'b+')
    plot(non_exc_odd(1:ceil(length(non_exc_odd)/2)),db(U(non_exc_odd(1:ceil(length(non_exc_odd)/2)))),'ro')
    plot(non_exc_even(1:ceil(length(non_exc_even)/2)),db(U(non_exc_even(1:ceil(length(non_exc_even)/2)))),'g*')
    xlabel("Frequency bins",'FontSize', 14,'interpreter','latex'); 
    ylabel("Input [dB]",'FontSize', 14,'interpreter','latex');
    legend("Exited odd harmonics","non-exited odd harmonics","non-exited even harmonics",...
        'FontSize', 14,'interpreter','latex')
    title('Odd multisine excitation','FontSize', 14,'interpreter','latex')
    xlim([0,length(uest)/2]), % if system == 4,ylim([-5,70]),end
%     xlim([0,length(uest)/2]),

    subplot(2,2,4); hold on;
    plot(excP(1:ceil(length(excP)/2)),db(Ym(excP(1:ceil(length(excP)/2)))),'b+')
    plot(noddP(1:ceil(length(noddP)/2)),db(Ym(noddP(1:ceil(length(noddP)/2)))),'ro')
    plot(nevenP(1:ceil(length(nevenP)/2)),db(Ym(nevenP(1:ceil(length(nevenP)/2)))),'g*')
    xlabel("Frequency bins",'FontSize', 14,'interpreter','latex'); 
    ylabel("Output [dB]",'FontSize', 14,'interpreter','latex');
    legend("Exited odd harmonics","non-exited odd harmonics","non-exited even harmonics",...
        'FontSize', 14,'interpreter','latex')
    xlim([0,length(yest)/2]), if system == 4,ylim([-50,70]),end
%     xlim([0,N/2]),
else
    if inputtype == 1
        noddP = (non_exc_odd - 1)*P + 1;
        nevenP = (non_exc_even - 1)*P + 1;
        subplot(2,2,2); hold on;
        plot(lines,db(U(excind(1:ceil(length(excind)/2)))),'b+')
        plot(non_exc_odd,db(U(non_exc_odd)),'ro')
        plot(non_exc_even,db(U(non_exc_even)),'g*')
        xlabel("Frequency bins",'FontSize', 14,'interpreter','latex'); 
        ylabel("Input [dB]",'FontSize', 14,'interpreter','latex');
        legend("Exited odd harmonics","non-exited odd harmonics","non-exited even harmonics",...
            'FontSize', 14,'interpreter','latex')
        title('Odd multisine excitation','FontSize', 14,'interpreter','latex')
        xlim([0,N/2])
        subplot(2,2,4); hold on;
        plot(excP(1:ceil(length(excP)/2)),db(Ym(excP(1:ceil(length(excP)/2)))),'b+')
        plot(noddP,db(Ym(noddP)),'ro')
        plot(nevenP,db(Ym(nevenP)),'g*')
        xlabel("Frequency bins",'FontSize', 14,'interpreter','latex'); 
        ylabel("Output [dB]",'FontSize', 14,'interpreter','latex');
        legend("Exited odd harmonics","non-exited odd harmonics","non-exited even harmonics",...
            'FontSize', 14,'interpreter','latex')
        xlim([0,N*P/2])
    else
        subplot(2,2,2); hold on;
        plot(db(U(1:ceil(length(U)/2))),'b+')
        xlabel("Frequency bins",'FontSize', 14,'interpreter','latex'); 
        ylabel("Input [dB]",'FontSize', 14,'interpreter','latex');
        legend("Gaussian input",'FontSize', 14,'interpreter','latex')
        title('Random excitation','FontSize', 14,'interpreter','latex')
        xlim([0,N/2])
        subplot(2,2,4); hold on;
        plot(db(Ym(1:ceil(length(Ym)/2))),'b+')
        xlabel("Frequency bins",'FontSize', 14,'interpreter','latex'); 
        ylabel("Output [dB]",'FontSize', 14,'interpreter','latex');
        legend("Output",'FontSize', 14,'interpreter','latex')
        xlim([0,N/2])
    end
end
end

noise = zeros(Npp,P);
Output = y_meas;
if SNR ~= Inf
    noise_power = mean(Output.^2)/(10^(SNR/10));
    for iter = 1:Nruns
        for p = 1:P
            noise(:,p,iter) = sqrt(noise_power) * randn(size(Output((p-1)*Npp+1:p*Npp)));
            y_meas((p-1)*Npp+1:p*Npp) = Output((p-1)*Npp+1:p*Npp) + noise(:,p,iter);
        end
    end
end

if inputnoise == 1
noise1 = zeros(Npp,P);
Input = u_meas;
if SNR ~= Inf
    noise_power = mean(Input.^2)/(10^(SNR/10));
    for iter = 1:Nruns
        for p = 1:P
            noise1(:,p,iter) = sqrt(noise_power) * randn(size(Input((p-1)*Npp+1:p*Npp)));
            u_meas((p-1)*Npp+1:p*Npp) = Input((p-1)*Npp+1:p*Npp) + noise1(:,p,iter);
        end
    end
end
end

if ~ismember(system,4:7)
    Output = reshape(Output,[N,R,NInputs,P]); Output = permute(Output,[1 4 2 3]);
    oest = Output(:,1:4);
    oval = Output(:,5);
    otest = Output(:,6);
    clear yest ytest yval
end

clear uexc t u us y y_meas Ym U pos non_exc_odd non_exc_even noddP nevenP nf excP excind plot_data

%% Volterra kernel estimation and PNLSS parameters extraction

Memory = MemoryV;
topt = zeros(Nruns,1);

for iter = 1:Nruns
    disp(['Volterra Monte-Carlo trial ',num2str(iter)])
    if ~ismember(system,4:7)
        yest = oest + noise(:,1:4,iter);
        yval = oval + noise(:,5,iter);
        ytest = otest + noise(:,6,iter);
        if inputnoise == 1
            uest = uest + noise1(:,1:4,iter);
            uval = uval + noise1(:,5,iter);
            utest = utest + noise1(:,6,iter);
        end
    else
        oest = yest;
        oval = yval;
        otest = ytest;
        if system == 4
            yest = oest + noise(1:100000);
            yval = oval + noise(100001:105000);
            ytest = otest + noise(105001:end);
        elseif system == 5
            yest = oest + noise(1:3000,1,iter);
            yval = oval + noise(3001:3500,1,iter);
            ytest = otest + noise(3501:end,1,iter);
        elseif system == 6
            yest = oest + noise(1:5500,1,iter);
            yval = oval + noise(5501:6500,1,iter);
            ytest = otest + noise(6501:end,1,iter);
        elseif system == 7
            yest = oest + noise(1:length(oest));
            yval = oval + noise(length(oest)+1:length([oest;oval]));
            ytest = otest + noise(length([oest;oval])+1:end);
        end
    end
    tic
    Volterrakernelidentification
    VolterraWKernelIdn
    if vopt == 0, topt(iter) = toc; end
    u = reshape(uest,[],1); y = reshape(yest,[],1); o = reshape(oest,[],1);
    OptimizationVolterra % Not timed, because optimization is computing output and plotting, no Optimization!
    if vopt == 1, topt(iter) = toc; end

    % Parameters of Block-oriented model
    if system == 1
        options = optimoptions(@lsqcurvefit,'Display','iter','Algorithm','levenberg-marquardt',...
        'FunValCheck','on','MaxFunctionEvaluations',25000,'MaxIter',25000,'TolFun',1e-16,'TolX',1e-16);%,'StepTolerance',1e-10);
        
        if simulation == 1
            [H1(:,iter),~] = freqz(cw',1,512);
            [be1(iter,:), ae1(iter,:)] = invfreqz(H1(:,iter),w,nb,na); % Use invfreqz to estimate b and a

            ltw = [pnl'; cw'];
            ynl_ae(:,iter) = filter(ltw(D1+1:end),1,uest(:)).^(1:D1)*ltw(1:D1);
            ynl_at(:,iter) = filter(ltw(D1+1:end),1,utest).^(1:D1)*ltw(1:D1);
            roest = oest(:); rnoest = yest(:);
            rmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
            rmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
            rrmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
            rrmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));
    
            rmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end)));
            rmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end)));
            rrmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
            rrmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));
            toc % Volterra + BS time


            ltw = [pnl'; zeros(cf_ord-length(pnl),1); cw'; zeros(1,length(coeff)-length(cw))'];
            [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(cf_ord+1:end),1,u).^(1:cf_ord)*h(1:cf_ord),...
            ltw,uest(:),yest(:),[],[],options);

            [H2(:,iter),~] = freqz(tmp(cf_ord+1:end),1,512);
            [be2(iter,:), ae2(iter,:)] = invfreqz(H2(:,iter),w,nb,na); % Use invfreqz to estimate b and a
            [H3(:,iter),~] = freqz(tmp(cf_ord+1:end)./norm(tmp(cf_ord+1:end)),1,512);
            [be3(iter,:), ae3(iter,:)] = invfreqz(H3(:,iter),w,nb,na); % Use invfreqz to estimate b and a
            tcf(iter) = toc; % time for curve fit.
    
            ynle(:,iter) = filter(be2(iter,:),ae2(iter,:),uest(:)).^(1:cf_ord)*tmp(1:cf_ord);
            ynl(:,iter) = filter(be2(iter,:),ae2(iter,:),utest).^(1:cf_ord)*tmp(1:cf_ord);
            
            prn_volt(iter) = 100*sqrt(mean((abs(Hn-H1(:,iter))).^2))/sqrt(sum(abs(Hn).^2));
            prn_cf(iter) = 100*sqrt(mean((abs(Hn-H3(:,iter))).^2))/sqrt(sum(abs(Hn).^2));

        else
            ltw = [pnl'; cw'];
            ynl_ae(:,iter) = filter(ltw(D1+1:end),1,uest(:)).^(1:D1)*ltw(1:D1);
            ynl_at(:,iter) = filter(ltw(D1+1:end),1,utest).^(1:D1)*ltw(1:D1);
    
            roest = oest(:); rnoest = yest(:);
            rmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
            rmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
            rrmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
            rrmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));
    
            rmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end)));
            rmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end)));
            rrmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
            rrmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));
            toc % Volterra + BS time

            ltw = [pnl'; zeros(cf_ord-length(pnl),1); cw'; zeros(1,ceil(1.0*length(cw)))'];
            [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(cf_ord+1:end),1,u).^(1:cf_ord)*h(1:cf_ord),...
            ltw,uest(:),yest(:),[],[],options);

            tcf(iter) = toc; % time for curve fit.
            
            ynle(:,iter) = filter(tmp(cf_ord+1:end),1,uest(:)).^(1:cf_ord)*tmp(1:cf_ord);
            ynl(:,iter) = filter(tmp(cf_ord+1:end),1,utest).^(1:cf_ord)*tmp(1:cf_ord);
        end
        valerr = otest - ynl(:,iter); roest = oest(:); rnoest = yest(:);
        rmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
        rmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
        rrmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
        rrmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));

        rmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),otest(length(ltw)+1:end)));
        rmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),ytest(length(ltw)+1:end)));
        rrmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
        rrmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));

        mu_cf(iter) = sum(valerr(length(ltw)+1:end))/(numel(valerr(length(ltw)+1:end)));
        s_cf(iter) = sqrt(sum(valerr(length(ltw)+1:end)-mu_cf(iter))^2/(numel(valerr(length(ltw)+1:end))));

        if iter == 1
            figure(8); hold on;
            plot(mean(ynl_at(:,iter),2),'DisplayName','Volterra BS')
            plot(mean(ynl,2),'DisplayName','Volterra curve fitting')
         end

    elseif system == 2
        H_kernel(find(H_kernel == 0)) = 1e-16;
        rank1mat = reshape(H_kernel,[],D1);
        [u1,s1,v1] = svd(rank1mat);
        rank1mat = u1(:,1)*s1(1,1)*v1(:,1)';
        rat(:,1) = rank1mat(:,2)./rank1mat(:,1);
        rat(:,2) = rank1mat(:,3)./rank1mat(:,1);
        avgp = mean(rat);

        options = optimoptions(@lsqcurvefit,'Display','iter','Algorithm','levenberg-marquardt',...
        'FunValCheck','on','MaxFunctionEvaluations',25000,'MaxIter',25000,'TolFun',1e-16,'TolX',1e-16,'StepTolerance',1e-10);
        
        if simulation == 1
            [H1(:,iter),~] = freqz(vker,1,512);
            [be1(iter,:), ae1(iter,:)] = invfreqz(H1(:,iter),w,nb,na); % Use invfreqz to estimate b and a
            [H2(:,iter),~] = freqz(rank1mat(:,1),1,512);
            [be2(iter,:), ae2(iter,:)] = invfreqz(H2(:,iter),w,nb,na); % Use invfreqz to estimate b and a
            [H2n(:,iter),~] = freqz(rank1mat(:,1)./norm(rank1mat(:,1)),1,512);
            [be2n(iter,:), ae2n(iter,:)] = invfreqz(H2n(:,iter),w,nb,na); % Use invfreqz to estimate b and a
            tR1(iter) = toc; % time for rank 1 approx.
            ynle1(:,iter) = filter(be2(iter,:),ae2(iter,:),uest(:) + avgp(1)*uest(:).^2 + avgp(2)*uest(:).^3);
            ynl1(:,iter) = filter(be2(iter,:),ae2(iter,:),utest(:) + avgp(1)*utest(:).^2 + avgp(2)*utest(:).^3);

            lth = [1 avgp zeros(1,cf_ord-length(avgp)-1) rank1mat(:,1)' zeros(1,length(coeff)-length(rank1mat(:,1)))];
            [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(cf_ord+1:end),1,(u.^(1:cf_ord))*h(1:cf_ord)'),...
                lth,uest(:),yest(:),[],[],options);

            [H3(:,iter),~] = freqz(tmp(cf_ord+1:end),1,512);
            [be3(iter,:), ae3(iter,:)] = invfreqz(H3(:,iter),w,nb,na); % Use invfreqz to estimate b and a
            [H4(:,iter),~] = freqz(tmp(cf_ord+1:end)./norm(tmp(cf_ord+1:end)),1,512);
            [be4(iter,:), ae4(iter,:)] = invfreqz(H4(:,iter),w,nb,na); % Use invfreqz to estimate b and a
            tcf(iter) = toc; % time for curve fit.

            regimp1 = [H_kernel(1:M+1)' zeros(1,length(coeff)-M-1)]; regimp4 = regimp1./norm(regimp1);
            regimp6 = [rank1mat(:,1)' zeros(1,length(coeff)-M-1)];
            regimp2 = regimp6./norm(regimp6);
            regimp3 = tmp(cf_ord+1:end); regimp5 = regimp3./norm(regimp3);
    
            ynle2(:,iter) = filter(be3(iter,:),ae3(iter,:),uest(:).^(1:cf_ord)*tmp(1:cf_ord)');
            ynl2(:,iter) = filter(be3(iter,:),ae3(iter,:),utest.^(1:cf_ord)*tmp(1:cf_ord)');

            prn_volt(iter) = 100*sqrt(mean((abs(Hn-H1(:,iter))).^2))/sqrt(sum(abs(Hn).^2));
            prn_R1T(iter) = 100*sqrt(mean((abs(Hn-H2n(:,iter))).^2))/sqrt(sum(abs(Hn).^2));
            prn_cf(iter) = 100*sqrt(mean((abs(Hn-H4(:,iter))).^2))/sqrt(sum(abs(Hn).^2));

        else
            tR1(iter) = toc; % time for rank 1 approx.
            ynle1(:,iter) = filter(rank1mat(:,1),1,uest(:).^(1:D1)*[1 avgp]');
            ynl1(:,iter) = filter(rank1mat(:,1),1,utest(:).^(1:D1)*[1 avgp]');

            lth = [1 avgp zeros(1,cf_ord-length(avgp)-1) rank1mat(:,1)' zeros(1,ceil(1.0*length(rank1mat(:,1))))];    
            [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(cf_ord+1:end),1,(u.^(1:cf_ord))*h(1:cf_ord)'),...
                lth,uest(:),yest(:),[],[],options);
            tcf(iter) = toc; % time for curve fit.

            ynle2(:,iter) = filter(tmp(cf_ord+1:end),1,uest(:).^(1:cf_ord)*tmp(1:cf_ord)');
            ynl2(:,iter) = filter(tmp(cf_ord+1:end),1,utest.^(1:cf_ord)*tmp(1:cf_ord)');
        end

        valerr = otest - ynl1(:,iter); roest = oest(:);
        rrmseerrR1T(iter) = sqrt(immse(ynle1(Memory+1:length(uest(:)),iter),roest(Memory+1:end))/sumsqr(roest(Memory+1:end)));
        rrmsterrR1T(iter) = sqrt(immse(otest(Memory+1:end),ynl1(Memory+1:length(utest),iter))/sumsqr(otest(Memory+1:end)));
        mu_R1T(iter) = sum(valerr(Memory+1:end))/(numel(valerr(Memory+1:end)));
        s_R1T(iter) = sqrt(sum(valerr(Memory+1:end)-mu_R1T(iter))^2/(numel(valerr(Memory+1:end))));

        valerr = otest - ynl2(:,iter);
        rrmseerrcf(iter) = sqrt(immse(ynle2(length(lth)-cf_ord+1:length(uest(:)),iter),roest(length(lth)-cf_ord+1:end))/sumsqr(roest(length(lth)-cf_ord+1:end)));
        rrmsterrcf(iter) = sqrt(immse(ynl2(length(lth)-cf_ord+1:length(utest),iter),otest(length(lth)-cf_ord+1:end))/sumsqr(otest(length(lth)-cf_ord+1:end)));
        mu_cf(iter) = sum(valerr(length(lth)-cf_ord+1:end))/(numel(valerr(length(lth)-cf_ord+1:end)));
        s_cf(iter) = sqrt(sum(valerr(length(lth)-cf_ord+1:end)-mu_cf(iter))^2/(numel(valerr(length(lth)-cf_ord+1:end))));

        if iter == Nruns
            figure(8); hold on; 
            plot(mean(ynl1,2),'DisplayName','Volterra rank-1 approx.')
            plot(mean(ynl2,2),'DisplayName','Volterra curve fitting')
        end

    elseif system == 3
        options = optimoptions(@lsqcurvefit,'Display','iter','Algorithm','levenberg-marquardt',...
        'FunValCheck','on','MaxFunctionEvaluations',25000,'MaxIter',25000,'TolFun',1e-16,'TolX',1e-16,'StepTolerance',1e-26);
        %PlotFcn={'optimplotx','optimplotfunccount','optimplotfval','optimplotresnorm','optimplotstepsize','optimplotfirstorderopt'});

        if simulation == 1
            if nlinearity == 1   
                pnlt = [coeff1(1)*coeff2(1)/(ncoeff1(1)*ncoeff2(1)) 0.4*coeff1(1)^2*coeff2(1)/(ncoeff1(1)^2*ncoeff2(1)) 0.2*coeff1(1)^3*coeff2(1)/(ncoeff1(1)^3*ncoeff2(1))];
                figure, hold on, plot([coeff1'./norm(coeff1) pnlt coeff2'./norm(coeff2)],'*')
            else
                figure, hold on, plot([coeff1'./norm(coeff1) coeff2'./norm(coeff2)],'*')
            end

            [H1(:,iter),~] = freqz(H_kernel(1:M+1),1,512);
            [be1(iter,:), ae1(iter,:)] = invfreqz(H1(:,iter),w,nb,na); % Use invfreqz to estimate b and a
            [H2(:,iter),~] = freqz(cw,1,512);
            [be2(iter,:), ae2(iter,:)] = invfreqz(H2(:,iter),w,nb1,na1); % Use invfreqz to estimate b and a
            [H3(:,iter),~] = freqz(ch,1,512);
            [be3(iter,:), ae3(iter,:)] = invfreqz(H3(:,iter),w,nb2,na2); % Use invfreqz to estimate b and a

            prn_volt(iter,1) = 100*sqrt(mean((abs(H-H1(:,iter))).^2))/sqrt(sum(abs(H).^2));
            prn_volt(iter,2) = 100*sqrt(mean((abs(Hw-H2(:,iter))).^2))/sqrt(sum(abs(Hw).^2));
            prn_volt(iter,3) = 100*sqrt(mean((abs(Hh-H3(:,iter))).^2))/sqrt(sum(abs(Hh).^2));
            prn_volt(iter,4) = 100*sqrt(mean((trueconv(1:M+1)-H_kernel(1:M+1)).^2))/sqrt(sum(trueconv(1:M+1).^2));

            e1 = 100*sqrt(mean((coeff1(1:MemoryW+1)./norm(coeff1(1:MemoryW+1))-cw').^2))/sqrt(sum((coeff1(1:MemoryW+1)./norm(coeff1(1:MemoryW+1))).^2));
            e2 = 100*sqrt(mean((coeff2(1:MemoryH+1)./norm(coeff2(1:MemoryH+1))-ch').^2))/sqrt(sum((coeff2(1:MemoryH+1)./norm(coeff2(1:MemoryH+1))).^2));
            e3 = 100*sqrt(mean((coeff1(1:MemoryW+1)./norm(coeff1(1:MemoryW+1))+cw').^2))/sqrt(sum((coeff1(1:MemoryW+1)./norm(coeff1(1:MemoryW+1))).^2));
            e4 = 100*sqrt(mean((coeff2(1:MemoryH+1)./norm(coeff2(1:MemoryH+1))+ch').^2))/sqrt(sum((coeff2(1:MemoryH+1)./norm(coeff2(1:MemoryH+1))).^2));

            if e1 > e3
                prn_volt(iter,5) = e3;
                cw = -cw;
            else
                prn_volt(iter,5) = e1;
            end
            if e2 > e4
                prn_volt(iter,6) = e4;
                ch = -ch;
            else
                prn_volt(iter,6) = e2;
            end
            
            %cw = randn(size(cw)); ch = randn(size(ch)); pnl = randn(size(pnl));
            ltw = [cw'; zeros(1,length(coeff1)-MemoryW-1)'];
            lth = [ch'; zeros(1,length(coeff2)-MemoryH-1)'];
            tmp = [ltw; pnl'; zeros(1,cf_ord-length(pnl))'; lth];
            if nlinearity == 1     
                plot(tmp)
                ynl_ae(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end), 1,filter(tmp(1:length(ltw)), 1,uest(:)).^(1:cf_ord)*pnlt');
                ynl_at(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end), 1,filter(tmp(1:length(ltw)), 1,utest(:)).^(1:cf_ord)*pnlt');
                yenorm = filter(ncoeff2,1,filter(ncoeff1,1,uest(:)).^(1:3)*pnlt');
                ytnorm = filter(ncoeff2,1,filter(ncoeff1,1,utest(:)).^(1:3)*pnlt');
            elseif nlinearity == 2
                plot([ltw; lth])
                ynl_ae(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end), 1,dzone(filter(tmp(1:length(ltw)), 1,uest(:)),ul,ll));
                ynl_at(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end), 1,dzone(filter(tmp(1:length(ltw)), 1,utest(:)),ul,ll));
                yenorm = filter(ncoeff2,1,dzone(filter(ncoeff1,1,uest(:)),ul,ll));
                ytnorm = filter(ncoeff2,1,dzone(filter(ncoeff1,1,utest(:)),ul,ll));
            else
                plot([ltw; lth])
                ynl_ae(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end), 1, 2./(1+exp(-0.8*(filter(tmp(1:length(ltw)), 1,uest(:)))))-1);
                ynl_at(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end), 1, 2./(1+exp(-0.8*(filter(tmp(1:length(ltw)), 1,utest(:)))))-1);
                yenorm = filter(ncoeff2,1, 2./(1+exp(-0.8*(filter(ncoeff1,1,uest(:)))))-1);
                ytnorm = filter(ncoeff2,1, 2./(1+exp(-0.8*(filter(ncoeff1,1,utest(:)))))-1);
            end
            yenormn = yenorm + reshape(noise(:,1:4),[],1);
            ytnormn = ytnorm + noise(:,6);

            roest = yenorm(:); rnoest = yenormn(:);
            rmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+length(lth)+1:length(uest(:)),iter),roest(length(ltw)+length(lth)+1:end)));
            rmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+length(lth)+1:length(uest(:)),iter),rnoest(length(ltw)+length(lth)+1:end)));
            rrmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+length(lth)+1:length(uest(:)),iter),roest(length(ltw)+length(lth)+1:end))/sumsqr(roest(length(ltw)+length(lth)+1:end)));
            rrmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+length(lth)+1:length(uest(:)),iter),rnoest(length(ltw)+length(lth)+1:end))/sumsqr(rnoest(length(ltw)+length(lth)+1:end)));

            rmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+length(lth)+1:length(utest(:)),iter),ytnorm(length(ltw)+length(lth)+1:end)));
            rmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+length(lth)+1:length(utest(:)),iter),ytnormn(length(ltw)+length(lth)+1:end)));
            rrmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+length(lth)+1:length(utest(:)),iter),ytnorm(length(ltw)+length(lth)+1:end))/sumsqr(ytnorm(length(ltw)+length(lth)+1:end)));
            rrmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+length(lth)+1:length(utest(:)),iter),ytnormn(length(ltw)+length(lth)+1:end))/sumsqr(ytnormn(length(ltw)+length(lth)+1:end)));
            toc % Volterra + BS time

            valerr = otest - ynl_at(:,iter);
            tmpmean = valerr(length(ltw)+length(lth)+1:end);
            mu_bs(iter) = mean(tmpmean);
            s_bs(iter) = std(tmpmean);

            tic                       
            [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(length(ltw)+cf_ord+1:end),1,filter(h(1:length(ltw)),1,u).^(1:cf_ord)*h(length(ltw)+1:length(ltw)+cf_ord)),...
            [ltw; pnl'; zeros(1,cf_ord-length(pnl))'; lth],uest(:),yest(:),[],[],options);

            e1 = 100*sqrt(mean((abs(ncoeff1-tmp(1:length(ltw))./norm(tmp(1:length(ltw))))).^2))/sqrt(sum(abs(ncoeff1).^2));
            e2 = 100*sqrt(mean((abs(ncoeff2-tmp(length(ltw)+cf_ord+1:end)./norm(tmp(length(ltw)+cf_ord+1:end)))).^2))/sqrt(sum(abs(ncoeff2).^2));
            e3 = 100*sqrt(mean((abs(ncoeff1+tmp(1:length(ltw))./norm(tmp(1:length(ltw))))).^2))/sqrt(sum(abs(ncoeff1).^2));
            e4 = 100*sqrt(mean((abs(ncoeff2+tmp(length(ltw)+cf_ord+1:end)./norm(tmp(length(ltw)+cf_ord+1:end)))).^2))/sqrt(sum(abs(ncoeff2).^2));

            if e1 > e3
                prn_cf(iter,3) = e3;
                tmp(1:length(ltw)) = -tmp(1:length(ltw));
            else
                prn_cf(iter,3) = e1;
            end
            if e2 > e4
                prn_cf(iter,4) = e4;
                tmp(length(ltw)+cf_ord+1:end) = -tmp(length(ltw)+cf_ord+1:end);
            else
                prn_cf(iter,4) = e2;
            end

            [H4(:,iter),~] = freqz(tmp(1:length(ltw)),1,512);
            [H4n(:,iter),~] = freqz(tmp(1:length(ltw))./norm(tmp(1:length(ltw))),1,512);
            [be4(iter,:), ae4(iter,:)] = invfreqz(H4(:,iter),w,nb1,na1); % Use invfreqz to estimate b and a
            [be4n(iter,:), ae4n(iter,:)] = invfreqz(H4n(:,iter),w,nb1,na1);

            [H5(:,iter),~] = freqz(tmp(length(ltw)+cf_ord+1:end),1,512);
            [H5n(:,iter),~] = freqz(tmp(length(ltw)+cf_ord+1:end)./norm(tmp(length(ltw)+cf_ord+1:end)),1,512);
            [be5(iter,:), ae5(iter,:)] = invfreqz(H5(:,iter),w,nb2,na2); % Use invfreqz to estimate b and a          
            [be5n(iter,:), ae5n(iter,:)] = invfreqz(H5n(:,iter),w,nb2,na2);

            tcf(iter) = toc;
            ynle(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end), 1,filter(tmp(1:length(ltw)), 1,uest(:)).^(1:cf_ord)*tmp(length(ltw)+1:length(ltw)+cf_ord));
            ynl(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end), 1,filter(tmp(1:length(ltw)), 1,utest(:)).^(1:cf_ord)*tmp(length(ltw)+1:length(ltw)+cf_ord));

            if nlinearity == 1   
                plot([tmp(1:length(ltw))./norm(tmp(1:length(ltw))); ...
                    tmp(length(ltw)+1:length(ltw)+cf_ord);...
                    tmp(length(ltw)+cf_ord+1:end)./norm(tmp(length(ltw)+cf_ord+1:end))])
            else
                %figure(2), hold on, 
                plot([tmp(1:length(ltw))./norm(tmp(1:length(ltw))); ...
                    tmp(length(ltw)+cf_ord+1:end)./norm(tmp(length(ltw)+cf_ord+1:end))])
            end

            prn_cf(iter,1) = 100*sqrt(mean((abs(Hw-H4n(:,iter))).^2))/sqrt(sum(abs(Hw).^2));
            prn_cf(iter,2) = 100*sqrt(mean((abs(Hh-H5n(:,iter))).^2))/sqrt(sum(abs(Hh).^2));
        else
            
            ynl_ae(:,iter) = filter(ch, 1,filter(cw, 1,uest(:)).^(1:D1)*pnl');
            ynl_at(:,iter) = filter(ch, 1,filter(cw, 1,utest(:)).^(1:D1)*pnl');

            roest = oest(:); rnoest = yest(:);
            rmseerrae(iter) = sqrt(immse(ynl_ae(length(cw)+length(ch)+1:length(uest(:)),iter),roest(length(cw)+length(ch)+1:end)));
            rmseerraen(iter) = sqrt(immse(ynl_ae(length(cw)+length(ch)+1:length(uest(:)),iter),rnoest(length(cw)+length(ch)+1:end)));
            rrmseerrae(iter) = sqrt(immse(ynl_ae(length(cw)+length(ch)+1:length(uest(:)),iter),roest(length(cw)+length(ch)+1:end))/sumsqr(roest(length(cw)+length(ch)+1:end)));
            rrmseerraen(iter) = sqrt(immse(ynl_ae(length(cw)+length(ch)+1:length(uest(:)),iter),rnoest(length(cw)+length(ch)+1:end))/sumsqr(rnoest(length(cw)+length(ch)+1:end)));

            rmsterrae(iter) = sqrt(immse(ynl_at(length(cw)+length(ch)+1:length(utest(:)),iter),otest(length(cw)+length(ch)+1:end)));
            rmsterraen(iter) = sqrt(immse(ynl_at(length(cw)+length(ch)+1:length(utest(:)),iter),ytest(length(cw)+length(ch)+1:end)));
            rrmsterrae(iter) = sqrt(immse(ynl_at(length(cw)+length(ch)+1:length(utest(:)),iter),otest(length(cw)+length(ch)+1:end))/sumsqr(otest(length(cw)+length(ch)+1:end)));
            rrmsterraen(iter) = sqrt(immse(ynl_at(length(cw)+length(ch)+1:length(utest(:)),iter),ytest(length(cw)+length(ch)+1:end))/sumsqr(ytest(length(cw)+length(ch)+1:end)));
            toc % Volterra + BS time

            ltw = [cw'; zeros(1,ceil(1.2*(MemoryW+1)))'];
            lth = [ch'; zeros(1,ceil(1.2*(MemoryH+1)))'];
            [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(length(ltw)+cf_ord+1:end),1,filter(h(1:length(ltw)),1,u).^(1:cf_ord)*h(length(ltw)+1:length(ltw)+cf_ord)),...
            [ltw; pnl'; zeros(1,cf_ord-length(pnl))'; lth],uest(:),yest(:),[],[],options);

            tcf(iter) = toc;
            
            ynle(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end),1,filter(tmp(1:length(ltw)),1,uest(:)).^(1:cf_ord)*tmp(length(ltw)+1:length(ltw)+cf_ord));
            ynl(:,iter) = filter(tmp(length(ltw)+cf_ord+1:end),1,filter(tmp(1:length(ltw)),1,utest(:)).^(1:cf_ord)*tmp(length(ltw)+1:length(ltw)+cf_ord));
        end

        roest = oest(:); rnoest = yest(:);
        rmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+length(lth)+1:length(uest(:)),iter),roest(length(ltw)+length(lth)+1:end)));
        rmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+length(lth)+1:length(uest(:)),iter),rnoest(length(ltw)+length(lth)+1:end)));
        rrmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+length(lth)+1:length(uest(:)),iter),roest(length(ltw)+length(lth)+1:end))/sumsqr(roest(length(ltw)+length(lth)+1:end)));
        rrmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+length(lth)+1:length(uest(:)),iter),rnoest(length(ltw)+length(lth)+1:end))/sumsqr(rnoest(length(ltw)+length(lth)+1:end)));

        rmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+length(lth)+1:length(utest(:)),iter),otest(length(ltw)+length(lth)+1:end)));
        rmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+length(lth)+1:length(utest(:)),iter),ytest(length(ltw)+length(lth)+1:end)));
        rrmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+length(lth)+1:length(utest(:)),iter),otest(length(ltw)+length(lth)+1:end))/sumsqr(otest(length(ltw)+length(lth)+1:end)));
        rrmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+length(lth)+1:length(utest(:)),iter),ytest(length(ltw)+length(lth)+1:end))/sumsqr(ytest(length(ltw)+length(lth)+1:end)));

        valerr = otest - ynl(:,iter);
        tmpmean = valerr(length(ltw)+length(lth)+1:end);
        mu_cf(iter) = mean(tmpmean);
        s_cf(iter) = std(tmpmean);

        if iter == Nruns
            figure(8); hold on;
            plot(mean(ynl,2),'DisplayName','Volterra curve fitting')
        end
    elseif system == 4
        options = optimoptions(@lsqcurvefit,'Display','iter','Algorithm','levenberg-marquardt',...
        'FunValCheck','on','MaxFunctionEvaluations',25000,'MaxIter',25000,'TolFun',1e-16,'TolX',1e-16,'StepTolerance',1e-26);

        ynl_ae(:,iter) = filter(ch, 1,filter(cw, 1,uest(:)).^(1:D1)*pnl');
        ynl_at(:,iter) = filter(ch, 1,filter(cw, 1,utest(:)).^(1:D1)*pnl');

        rmseerrae = sqrt(sum((yest(5051:end)-ynl_ae(5051:end)).^2)/94950);
        rrmseerrae = sqrt((sum((yest(5051:end)-ynl_ae(5051:end)).^2)/94950)/sumsqr(yest(5051:end)));
        rmsterrae = sqrt(sum((ytest(1001:79000)-ynl_at(1001:79000)).^2)/78000);
        rrmsterrae = sqrt((sum((ytest(1001:79000)-ynl_at(1001:79000)).^2)/78000)/sumsqr(ytest(1001:79000)));
        toc % Volterra + BS time
  
        %cw = randn(size(cw)); ch = randn(size(ch)); pnl = randn(size(pnl));
        ltw = [cw'; zeros(1,ceil(2.0*(MemoryW+1)))'];
        lth = [ch'; zeros(1,ceil(2.0*(MemoryH+1)))'];
        tmp = [ltw; pnl'; zeros(1,cf_ord-length(pnl))'; lth];
        ynle = filter(tmp(length(ltw)+cf_ord+1:end),1,filter(tmp(1:length(ltw)),1,uest(:)).^(1:cf_ord)*tmp(length(ltw)+1:length(ltw)+cf_ord));
        ynl = filter(tmp(length(ltw)+cf_ord+1:end),1,filter(tmp(1:length(ltw)),1,utest(:)).^(1:cf_ord)*tmp(length(ltw)+1:length(ltw)+cf_ord));
        initialerr = [1000*sqrt(sum((yest(5051:end)-ynle(5051:end)).^2)/94950),...
            1000*sqrt(sum((ytest(1001:79000)-ynl(1001:79000)).^2)/78000),...
            100*sqrt((sum((yest(5051:end)-ynle(5051:end)).^2)/94950)/sumsqr(yest(5051:end))),...
            100*sqrt((sum((ytest(1001:79000)-ynl(1001:79000)).^2)/78000)/sumsqr(ytest(1001:79000)))];

        ie = [1000*rmseerrae,...
            1000*rmsterrae,...
            100*rrmseerrae,...
            100*rrmsterrae];

        [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(length(ltw)+cf_ord+1:end),1,filter(h(1:length(ltw)),1,u).^(1:cf_ord)*h(length(ltw)+1:length(ltw)+cf_ord)),...
        [ltw; pnl'; zeros(1,cf_ord-length(pnl))'; lth],uest(:),yest(:),[],[],options);

        tcf = toc;
        
        ynle = filter(tmp(length(ltw)+cf_ord+1:end),1,filter(tmp(1:length(ltw)),1,uest(:)).^(1:cf_ord)*tmp(length(ltw)+1:length(ltw)+cf_ord));
        ynl = filter(tmp(length(ltw)+cf_ord+1:end),1,filter(tmp(1:length(ltw)),1,utest(:)).^(1:cf_ord)*tmp(length(ltw)+1:length(ltw)+cf_ord));
        
        valerr = ytest - ynl; %roest = oest(:);
        rmseerrcf = sqrt(sum((yest(1001:end)-ynle(1001:end)).^2)/99000);
        rrmseerrcf = sqrt((sum((yest(1001:end)-ynle(1001:end)).^2)/99000)/sumsqr(yest(1001:end)));
        rmsterrcf = sqrt(sum((ytest(1001:79000)-ynl(1001:79000)).^2)/78000);
        rrmsterrcf = sqrt((sum((ytest(1001:79000)-ynl(1001:79000)).^2)/78000)/sumsqr(ytest(1001:79000)));
        mu_cf = sum(valerr(1001:79000))/78000;
        s_cf = sqrt(sum((valerr(1001:79000)-mu_cf).^2)/78000);

        figure(8); hold on;
        plot(ynl_at,'DisplayName','Volterra BS')
        plot(ynl,'DisplayName','Volterra curve fitting')
        
    elseif system == 5
        options = optimoptions(@lsqcurvefit,'Display','iter','Algorithm','levenberg-marquardt',...
        'FunValCheck','on','MaxFunctionEvaluations',25000,'MaxIter',25000,'TolFun',1e-16,'TolX',1e-16);%,'StepTolerance',1e-10);
        %cf_ord = 9; vker = randn(length(vker),1); pnl = randn(length(pnl),1)';

        ltw = [pnl'; cw'];
        ynl_ae(:,iter) = filter(ltw(D1+1:end),1,uest(:)).^(1:D1)*ltw(1:D1);
        ynl_at(:,iter) = filter(ltw(D1+1:end),1,utest).^(1:D1)*ltw(1:D1);

        roest = oest(:); rnoest = yest(:);
        rmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
        rmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
        rrmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
        rrmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));

        rmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end)));
        rmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end)));
        rrmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
        rrmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));
        toc % Volterra + BS time

        ltw = [pnl'; zeros(cf_ord-length(pnl),1); cw'; zeros(1,ceil(1.0*length(cw)))'];
        [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(cf_ord+1:end),1,u).^(1:cf_ord)*h(1:cf_ord),...
            ltw,uest(:),yest(:),[],[],options);

        tcf(iter) = toc; % time for curve fit.
         
        ynle(:,iter) = filter(tmp(cf_ord+1:end),1,uest(:)).^(1:cf_ord)*tmp(1:cf_ord);
        ynl(:,iter) = filter(tmp(cf_ord+1:end),1,utest).^(1:cf_ord)*tmp(1:cf_ord);

        valerr = otest - ynl(:,iter); roest = oest(:); rnoest = yest(:);
        rmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
        rmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
        rrmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
        rrmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));

        rmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),otest(length(ltw)+1:end)));
        rmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),ytest(length(ltw)+1:end)));
        rrmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
        rrmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));

        mu_cf(iter) = sum(valerr(length(ltw)+1:end))/(numel(valerr(length(ltw)+1:end)));
        s_cf(iter) = sqrt(sum(valerr(length(ltw)+1:end)-mu_cf(iter))^2/(numel(valerr(length(ltw)+1:end))));

        if iter == 1
            figure(8); hold on;
            plot(ynl_at(:,iter),'DisplayName','Volterra BS')
            plot(ynl(:,iter),'DisplayName','Volterra curve fitting')
        end
    elseif system == 6
        options = optimoptions(@lsqcurvefit,'Display','iter','Algorithm','levenberg-marquardt',...
        'FunValCheck','on','MaxFunctionEvaluations',25000,'MaxIter',25000,'TolFun',1e-16,'TolX',1e-16);%,'StepTolerance',1e-10);
        
        ltw = [pnl'; cw';];
        ynl_ae(:,iter) = filter(ltw(D1+1:end),1,uest(:)).^(1:D1)*ltw(1:D1);
        ynl_at(:,iter) = filter(ltw(D1+1:end),1,utest).^(1:D1)*ltw(1:D1);

        roest = oest(:); rnoest = yest(:);
        rmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
        rmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
        rrmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
        rrmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));

        rmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end)));
        rmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end)));
        rrmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
        rrmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));
        toc % Volterra + BS time

        ltw = [pnl'; zeros(cf_ord-length(pnl),1); cw'; zeros(1,ceil(2.0*length(cw)))'];
        [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(cf_ord+1:end),1,u).^(1:cf_ord)*h(1:cf_ord),...
        ltw,uest(:),oest(:),[],[],options);

        tcf = toc; % time for curve fit.
        
        ynle(:,iter) = filter(tmp(cf_ord+1:end),1,uest(:)).^(1:cf_ord)*tmp(1:cf_ord);
        ynl(:,iter) = filter(tmp(cf_ord+1:end),1,utest).^(1:cf_ord)*tmp(1:cf_ord);

        valerr = otest - ynl(:,iter); roest = oest(:); rnoest = yest(:);
        rmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
        rmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
        rrmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
        rrmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));

        rmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),otest(length(ltw)+1:end)));
        rmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),ytest(length(ltw)+1:end)));
        rrmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
        rrmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));

        mu_cf(iter) = sum(valerr(length(ltw)+1:end))/(numel(valerr(length(ltw)+1:end)));
        s_cf(iter) = sqrt(sum(valerr(length(ltw)+1:end)-mu_cf(iter))^2/(numel(valerr(length(ltw)+1:end))));

        if iter == Nruns
            figure(8); hold on;
            plot(ynl_at(:,iter),'DisplayName','Volterra BS')
            plot(ynl(:,iter),'DisplayName','Volterra curve fitting')
        end
    elseif system == 7
        options = optimoptions(@lsqcurvefit,'Display','iter','Algorithm','levenberg-marquardt',...
        'FunValCheck','on','MaxFunctionEvaluations',25000,'MaxIter',25000,'TolFun',1e-16,'TolX',1e-16);%,'StepTolerance',1e-10);
%         cf_ord = 3; %vker = randn(length(vker),1); pnl = randn(length(pnl),1)';
        ltw = [pnl'; cw'];

        ynl_ae(:,iter) = filter(ltw(D1+1:end),1,uest(:)).^(1:D1)*ltw(1:D1);
        ynl_at(:,iter) = filter(ltw(D1+1:end),1,utest).^(1:D1)*ltw(1:D1);

        roest = oest(:); rnoest = yest(:);
        rmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
        rmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
        rrmseerrae(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
        rrmseerraen(iter) = sqrt(immse(ynl_ae(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));

        rmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end)));
        rmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end)));
        rrmsterrae(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
        rrmsterraen(iter) = sqrt(immse(ynl_at(length(ltw)+1:length(utest(:)),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));
        toc % Volterra + BS time

        ltw = [pnl'; zeros(cf_ord-length(pnl),1); cw'; zeros(1,ceil(1.0*length(cw)))'];
        [tmp,resnorm,~,~,~,~,jacobian] = lsqcurvefit(@(h,u)filter(h(cf_ord+1:end),1,u).^(1:cf_ord)*h(1:cf_ord),...
            ltw,uest(:),yest(:),[],[],options);

        tcf(iter) = toc; % time for curve fit.

        ynle(:,iter) = filter(tmp(cf_ord+1:end),1,uest(:)).^(1:cf_ord)*tmp(1:cf_ord);
        ynl(:,iter) = filter(tmp(cf_ord+1:end),1,utest).^(1:cf_ord)*tmp(1:cf_ord);

        valerr = otest - ynl(:,iter);
        rmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end)));
        rmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end)));
        rrmseerrcf(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),roest(length(ltw)+1:end))/sumsqr(roest(length(ltw)+1:end)));
        rrmseerrcfn(iter) = sqrt(immse(ynle(length(ltw)+1:length(uest(:)),iter),rnoest(length(ltw)+1:end))/sumsqr(rnoest(length(ltw)+1:end)));

        rmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),otest(length(ltw)+1:end)));
        rmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),ytest(length(ltw)+1:end)));
        rrmsterrcf(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),otest(length(ltw)+1:end))/sumsqr(otest(length(ltw)+1:end)));
        rrmsterrcfn(iter) = sqrt(immse(ynl(length(ltw)+1:length(utest),iter),ytest(length(ltw)+1:end))/sumsqr(ytest(length(ltw)+1:end)));

        mu_cf(iter) = sum(valerr(length(ltw)+1:end))/(numel(valerr(length(ltw)+1:end)));
        s_cf(iter) = sqrt(sum((valerr(length(ltw)+1:end)-mu_cf(iter)).^2)/(numel(valerr(length(ltw)+1:end))));

        if iter == Nruns
            figure(8); hold on;
            plot(ynl_at,'DisplayName','Volterra BS')
            plot(ynl(:,iter),'DisplayName','Volterra curve fitting')
        end
    end
end

switch system
    case {1,5,6,7}
        elapsedTime(1) = sum(topt)/Nruns;
        elapsedTime(2) = sum(tcf)/Nruns;

        rrmseErr(1) = sum(rrmseerrVolt)/Nruns; perceErr(1) = 100*rrmseErr(1);
        rrmstErr(1) = sum(rrmsterrVolt)/Nruns; perctErr(1) = 100*rrmstErr(1);
        rrmseErrn(1) = sum(rrmseerrVoltn)/Nruns; perceErrn(1) = 100*rrmseErrn(1);
        rrmstErrn(1) = sum(rrmsterrVoltn)/Nruns; perctErrn(1) = 100*rrmstErrn(1);

        rmseErr(1) = sum(rmseerrVolt)/Nruns;
        rmstErr(1) = sum(rmsterrVolt)/Nruns; 
        rmseErrn(1) = sum(rmseerrVoltn)/Nruns;
        rmstErrn(1) = sum(rmsterrVoltn)/Nruns;

        rrmseErr(2) = sum(rrmseerrcf)/Nruns; perceErr(2) = 100*rrmseErr(2);
        rrmstErr(2) = sum(rrmsterrcf)/Nruns; perctErr(2) = 100*rrmstErr(2);
        rrmseErrn(2) = sum(rrmseerrcfn)/Nruns; perceErrn(2) = 100*rrmseErrn(2);
        rrmstErrn(2) = sum(rrmsterrcfn)/Nruns; perctErrn(2) = 100*rrmstErrn(2);

        rmseErr(2) = sum(rmseerrcf)/Nruns;
        rmstErr(2) = sum(rmsterrcf)/Nruns;
        rmseErrn(2) = sum(rmseerrcfn)/Nruns;
        rmstErrn(2) = sum(rmsterrcfn)/Nruns;

        Errae(1,:) = [sum(rmseerrae)/Nruns sum(rmseerraen)/Nruns sum(rmsterrae)/Nruns sum(rmsterraen)/Nruns...
                100*sum(rrmseerrae)/Nruns 100*sum(rrmseerraen)/Nruns 100*sum(rrmsterrae)/Nruns 100*sum(rrmsterraen)/Nruns ];

        Marr = [MemoryV length(tmp)-cf_ord+1];
        mu_t = [sum(mu_tr)/Nruns sum(mu_cf)/Nruns]; s_t = [sum(s_tr)/Nruns sum(s_cf)/Nruns];
        params = [length(H_kernel) length(tmp)];
        if simulation == 1
            prn = [sum(prn_volt)/Nruns sum(prn_cf)/Nruns];
            sys1 = tf(b,a,1); sys2 = tf(be1,ae1,1); sys3 = tf(be2,ae2,1);
            figure, bode(sys1,'b'), hold on, bode(sys2,'g'), bode(sys3,'k')
            legend('True system','Volterra linear kernel estimates','Volterra curve fitting'), 
            title('Bode plot of true and estimated LTI filter')
        end

    case 2
        elapsedTime(1) = sum(topt)/Nruns;
        elapsedTime(2) = sum(tR1)/Nruns;
        elapsedTime(3) = sum(tcf)/Nruns; 
        rrmseErr(1) = sum(rrmseerrVolt)/Nruns; perceErr(1) = 100*rrmseErr(1);
        rrmstErr(1) = sum(rrmsterrVolt)/Nruns; perctErr(1) = 100*rrmstErr(1);
        rrmseErr(2) = sum(rrmseerrR1T)/Nruns; perceErr(2) = 100*rrmseErr(2);
        rrmstErr(2) = sum(rrmsterrR1T)/Nruns; perctErr(2) = 100*rrmstErr(2);
        rrmseErr(3) = sum(rrmseerrcf)/Nruns; perceErr(3) = 100*rrmseErr(3);
        rrmstErr(3) = sum(rrmsterrcf)/Nruns; perctErr(3) = 100*rrmstErr(3);
        Marr(1) = MemoryV; Marr(2) = MemoryV; Marr(3) = length(tmp)-cf_ord-1;
        mu_t(1) = sum(mu_tr)/Nruns; s_t(1) = sum(s_tr)/Nruns;
        mu_t(2) = sum(mu_R1T)/Nruns; s_t(2) = sum(s_R1T)/Nruns;
        mu_t(3) = sum(mu_cf)/Nruns; s_t(3) = sum(s_cf)/Nruns;
        params(1) = length(H_kernel);
        params(2) = length(rank1mat(:,1)) + length(avgp);
        params(3) = length(tmp);
        if simulation == 1
            prn(1) = sum(prn_volt)/Nruns;
            prn(2) = sum(prn_R1T)/Nruns;
            prn(3) = sum(prn_cf)/Nruns;
            be1b = mean(be1,1); ae1b = mean(ae1,1);
            be2nb = mean(be2n,1); ae2nb = mean(ae2n,1);
            be4b = mean(be4,1); ae4b = mean(ae4,1);
            if any(abs(zero(tf(be1b./be1b(1),ae1b./ae1b(1),1)))<1)
                z = roots(be1b./be1b(1));
                alpha = 1.04; % Or whatever factor moves them just outside
                z_new = alpha * z;
                be1b = poly(z_new);
            end
            if any(abs(zero(tf(be2nb./be2nb(1),ae2nb./ae2nb(1),1)))<1)
                z = roots(be2nb./be2nb(1));
                alpha = 1.04; % Or whatever factor moves them just outside
                z_new = alpha * z;
                be2nb = poly(z_new);
            end
            if any(abs(zero(tf(be4b./be4b(1),ae4b./ae4b(1),1)))<1)
                z = roots(be4b./be4b(1));
                alpha = 1.001; % Or whatever factor moves them just outside
                z_new = alpha * z;
                be4b = poly(z_new);
            end
            sys1 = tf(b./b(1),a./a(1),1); sys2 = tf(be1b./be1b(1),ae1b./ae1b(1),1); 
            sys3 = tf(be2nb./be2nb(1),ae2nb./ae2nb(1),1); sys4 = tf(be4b./be4b(1),ae4b./ae4b(1),1);
            figure, bode(sys1,'-b',sys2,':g',sys3,'-.r',sys4,'--k')

            legend({'LTI Block - Ground trth','Prior-Informed Volterra linear kernel','BS Volterra','BS Volterra LM'},'interpreter','latex'), 
            set(gca,'TickLabelInterpreter','latex'); set(gca,'FontSize',10),
            title('Bode plot of true and estimated LTI filter - SNR 5 dB') %  
        end
    case 3
        elapsedTime = [sum(topt)/Nruns sum(tcf)/Nruns];

        rrmseErr(1) = sum(rrmseerrVolt)/Nruns; perceErr(1) = 100*rrmseErr(1);
        rrmstErr(1) = sum(rrmsterrVolt)/Nruns; perctErr(1) = 100*rrmstErr(1);
        rrmseErrn(1) = sum(rrmseerrVoltn)/Nruns; perceErrn(1) = 100*rrmseErrn(1);
        rrmstErrn(1) = sum(rrmsterrVoltn)/Nruns; perctErrn(1) = 100*rrmstErrn(1);

        rmseErr(1) = sum(rmseerrVolt)/Nruns;
        rmstErr(1) = sum(rmsterrVolt)/Nruns; 
        rmseErrn(1) = sum(rmseerrVoltn)/Nruns;
        rmstErrn(1) = sum(rmsterrVoltn)/Nruns;

        rrmseErr(2) = sum(rrmseerrcf)/Nruns; perceErr(2) = 100*rrmseErr(2);
        rrmstErr(2) = sum(rrmsterrcf)/Nruns; perctErr(2) = 100*rrmstErr(2);
        rrmseErrn(2) = sum(rrmseerrcfn)/Nruns; perceErrn(2) = 100*rrmseErrn(2);
        rrmstErrn(2) = sum(rrmsterrcfn)/Nruns; perctErrn(2) = 100*rrmstErrn(2);

        rmseErr(2) = sum(rmseerrcf)/Nruns;
        rmstErr(2) = sum(rmsterrcf)/Nruns;
        rmseErrn(2) = sum(rmseerrcfn)/Nruns;
        rmstErrn(2) = sum(rmsterrcfn)/Nruns;

        Errae(1,:) = [sum(rmseerrae)/Nruns sum(rmseerraen)/Nruns sum(rmsterrae)/Nruns sum(rmsterraen)/Nruns...
            100*sum(rrmseerrae)/Nruns 100*sum(rrmseerraen)/Nruns 100*sum(rrmsterrae)/Nruns 100*sum(rrmsterraen)/Nruns ];

        Marr = [MemoryV length(tmp)-cf_ord+1];
        mu_t = [mean(mu_tr) mean(mu_bs) mean(mu_cf)]; 
        s_t = [mean(s_tr) mean(s_bs) mean(s_cf)];
        params = [length(H_kernel) length(tmp)];
        if simulation == 1
            for i = 1:6
                prn(1,i) = sum(prn_volt(:,i))/Nruns;
            end
            for i = 1:4
                prn(2,i) = sum(prn_cf(:,i))/Nruns;
            end
        end
    case 4
        elapsedTime = [topt tcf];
        rmseErr = [rmseerrVolt rmseerrcf];
        rrmseErr = [rrmseerrVolt rrmseerrcf]; perceErr = 100*rrmseErr;
        rmstErr = [rmsterrVolt rmsterrcf];
        rrmstErr = [rrmsterrVolt rrmsterrcf]; perctErr = 100*rrmstErr;
        Marr = [MemoryV length(tmp)-cf_ord+1];
        mu_t = [mu_tr mu_cf]; s_t = [s_tr s_cf];
        params = [length(H_kernel) length(tmp)];
end

clear A B C D delta E estnlerr F fpos gamma h Ha Hvec I idx imp indices inds ker linestyle...
    marker nm normpar options q1 r rat s1 U u1 Ut Uv v1 valerr

% if Nruns == 40
%     Z = 2.02;
% elseif Nruns ==80
%     Z = 1.99;
% elseif Nruns == 120
%     Z = 1.98;
% end
% errors = [rrmsterrVolt' rrmsterrae' rrmsterrcf'];
% meanerrs = mean(errors);
% stderrs = std(errors.*100);
% CI = 2*Z*stderrs./sqrt(Nruns);
% 
% perrors = [prn_volt prn_cf];
% pmeanerrs = mean(perrors);
% pstderrs = std(perrors);
% pCI = 2*Z*pstderrs./sqrt(Nruns);

%% Linear estimation by classical approach and optimize for full PNLSS parameters

if SimAll == 1
    D1 = 3;
    true_model.nx = 2:D1;
    true_model.ny = 2:D1;
    Memory = Memoryl;
    if simulation == 1 && inputtype == 1 , lines = linestmp; 
    elseif ~ismember(system,4:7), lines = 1:length(uest); end
    tic
    uf = reshape(fft(uest),[size(uest,1),1,1,size(uest,2)]); uf = uf(lines,:,:,:);
    % frequ = (lines-1)/N; % Excited frequencies (normalized)
    for iter = 1:Nruns
        disp(['BLA Monte-Carlo trial ',num2str(iter)])
        if ~ismember(system,4:7)
            yest = oest + noise(:,1:4,iter);
            yval = oval + noise(:,5,iter);
            ytest = otest + noise(:,6,iter);
        else
            oest = yest;
            oval = yval;
            otest = ytest;
            if system == 4
                yest = oest + noise(1:100000);
                yval = oval + noise(100001:105000);
                ytest = otest + noise(105001:end);
            elseif system == 5
                yest = oest + noise(1:3000,1,iter);
                yval = oval + noise(3001:3500,1,iter);
                ytest = otest + noise(3501:end,1,iter);
            elseif system == 6
                yest = oest + noise(1:5500,1,iter);
                yval = oval + noise(5501:6500,1,iter);
                ytest = otest + noise(6501:end,1,iter);
            elseif system == 7
                yest = oest + noise(1:length(oest));
                yval = oval + noise(length(oest)+1:length([oest;oval]));
                ytest = otest + noise(length([oest;oval])+1:end);
            end
        end
        yf = reshape(fft(yest),[size(yest,1),1,1,size(yest,2)]); yf = yf(lines,:,:,:);
        covY = fCovarY(reshape(yest,[size(yest,1),1,1,size(yest,2)])); % Noise covariance (frequency domain)
        frequ = (lines-1)/N;
        % Frequency response function
        [G,covGML,covGn] = fCovarFrf(uf,yf);
        % figure(2); hold on,
        % plot(db([G(:) covGn(:)])) % covGML=0 if R=1, covGn=0 if P=1
        % xlabel('Frequency line'), ylabel('Amplitude (dB)')
        % legend('FRF','Noise distortion') %,'Total distortion'
        
        BLAestimate
        OptimizationBLA
    end
    if system == 2
        loopi = 4;
    else
        loopi = 3;
    end
    elapsedTime(loopi) = toc/Nruns;
    Marr(loopi) = Memoryl;
    if system == 4
        rmseErr = [rmseerrVolt rmseerrcf rmseerrBLA];
        rrmseErr = [rrmseerrVolt rrmseerrcf rrmseerrBLA]; perceErr = 100*rrmseErr;
        rmstErr = [rmsterrVolt rmsterrcf rmsterrBLA];
        rrmstErr = [rrmsterrVolt rrmsterrcf rrmsterrBLA]; perctErr = 100*rrmstErr;
    else
        rrmseErr(loopi) = sum(rrmseerrBLA)/Nruns; perceErr(loopi) = 100*rrmseErr(loopi);
        rrmstErr(loopi) = sum(rrmsterrBLA)/Nruns; perctErr(loopi) = 100*rrmstErr(loopi);
        rrmseErrn(loopi) = sum(rrmseerrBLAn)/Nruns; perceErrn(loopi) = 100*rrmseErrn(loopi);
        rrmstErrn(loopi) = sum(rrmsterrBLAn)/Nruns; perctErrn(loopi) = 100*rrmstErrn(loopi);
    
        rmseErr(loopi) = sum(rmseerrBLA)/Nruns;
        rmstErr(loopi) = sum(rmsterrBLA)/Nruns;
        rmseErrn(loopi) = sum(rmseerrBLAn)/Nruns;
        rmstErrn(loopi) = sum(rmsterrBLAn)/Nruns;
    end

    mu_t(loopi) = sum(mu_tr)/Nruns; s_t(loopi) = sum(s_tr)/Nruns;
    if system ~= 3 || system ~= 4
        params(loopi) = numel(bestmodelBLA(1).A)+numel(bestmodelBLA(1).B)+numel(bestmodelBLA(1).C)...
            +numel(bestmodelBLA(1).D)+nnz(bestmodelBLA(1).E)+nnz(bestmodelBLA(1).F);
    else
        params(loopi) = (nchoosek(Memory+NInputs+max(true_model.ny),max(true_model.ny))-1)*(Memory+NOutputs)-Memory^2;
    end

    clear A B C D covGML covY covGn ...
    G uf yf err err_lin esterr frequ freq i lines ...
    m maxr min_err min_valerr models model modellintest n na NTrans Nval ...
    plotfreq plottime states t T1 T2 tval u valerrs valerr W y y_lin y_mod...
    ytest_lin yval_mod yval_hat p
end
    
%% Linear estimation by subspace-based method
 
if SimAll == 1
    if simulation == 0, Memory = Memorys; end
    tic
    for iter = 1:Nruns
        disp(['N4SID Monte-Carlo trial ',num2str(iter)])
        if ~ismember(system,4:7)
            yest = oest + noise(:,1:4,iter);
            yval = oval + noise(:,5,iter);
            ytest = otest + noise(:,6,iter);
        else
            oest = yest;
            oval = yval;
            otest = ytest;
            if system == 4
                yest = oest + noise(1:100000);
                yval = oval + noise(100001:105000);
                ytest = otest + noise(105001:end);
            elseif system == 5
                yest = oest + noise(1:3000,1,iter);
                yval = oval + noise(3001:3500,1,iter);
                ytest = otest + noise(3501:end,1,iter);
            elseif system == 6
                yest = oest + noise(1:5500,1,iter);
                yval = oval + noise(5501:6500,1,iter);
                ytest = otest + noise(6501:end,1,iter);
            elseif system == 7
                yest = oest + noise(1:length(oest));
                yval = oval + noise(length(oest)+1:length([oest;oval]));
                ytest = otest + noise(length([oest;oval])+1:end);
            end
        end
        data = iddata(yest(:),uest(:),ts);
        opt = n4sidOptions('InitialState','zero','EnforceStability',1,'N4weight','CVA');
        sys = n4sid(data,Memory,opt,'Feedthrough',1,'DisturbanceModel','none','Form','canonical');
        A = sys.A; B = sys.B; C = sys.C; D = sys.D;
        u = reshape(uest,[],1);
        y = reshape(yest,[],1);
        o = reshape(oest,[],1);
        OptimizationN4SID
    end
    loopi = loopi+1;
    elapsedTime(loopi) = toc/Nruns;
    Marr(loopi) = Memorys;
    if system == 4
        rmseErr = [rmseerrVolt rmseerrcf rmseerrBLA rmseerrSID];
        rrmseErr = [rrmseerrVolt rrmseerrcf rrmseerrBLA rrmseerrSID]; perceErr = 100*rrmseErr;
        rmstErr = [rmsterrVolt rmsterrcf rmsterrBLA rmsterrSID];
        rrmstErr = [rrmsterrVolt rrmsterrcf rrmsterrBLA rrmsterrSID]; perctErr = 100*rrmstErr;
    else
        rrmseErr(loopi) = sum(rrmseerrSID)/Nruns; perceErr(loopi) = 100*rrmseErr(loopi);
        rrmstErr(loopi) = sum(rrmsterrSID)/Nruns; perctErr(loopi) = 100*rrmstErr(loopi);
        rrmseErrn(loopi) = sum(rrmseerrSIDn)/Nruns; perceErrn(loopi) = 100*rrmseErrn(loopi);
        rrmstErrn(loopi) = sum(rrmsterrSIDn)/Nruns; perctErrn(loopi) = 100*rrmstErrn(loopi);
    
        rmseErr(loopi) = sum(rmseerrSID)/Nruns;
        rmstErr(loopi) = sum(rmsterrSID)/Nruns;
        rmseErrn(loopi) = sum(rmseerrSIDn)/Nruns;
        rmstErrn(loopi) = sum(rmsterrSIDn)/Nruns;
    end

    mu_t(loopi) = sum(mu_tr)/Nruns; s_t(loopi) = sum(s_tr)/Nruns;
    if system ~= 3 || system ~= 4
        params(loopi) = numel(bestmodelSID(1).A)+numel(bestmodelSID(1).B)+numel(bestmodelSID(1).C)...
            +numel(bestmodelSID(1).D)+nnz(bestmodelSID(1).E)+nnz(bestmodelSID(1).F);
    else
        params(loopi) = numel(bestmodelSID(1).A)+numel(bestmodelSID(1).B)+numel(bestmodelSID(1).C)...
            +numel(bestmodelSID(1).D)+numel(bestmodelSID(1).E)+numel(bestmodelSID(1).F);
    end
%     if simulation == 1
%         if system == 3
%             prn(5,1) = sum(prn_sid)/Nruns;
%         else
%             prn(loopi) = sum(prn_sid)/Nruns;
%         end
%     end
end


%% Save figs and results

% FigList = findobj(allchild(0),'flat','Type','figure');
% for iFig = 1:length(FigList)
%     FigHandle = FigList(iFig);
%     FigName   = num2str(get(FigHandle, 'Number'));
%     set(0, 'CurrentFigure', FigHandle);
%     savefig(fullfile([FigName, '.fig']));    %<---- 'Brackets'
% end

if simulation == 1
%     if nlinearity == 1
%         save('results.mat','mu_t','s_t','rmseErr','rmseErrn','rmstErr','rmstErrn','perceErr','perceErrn','perctErr','perctErrn','elapsedTime','params','prn','Marr','Errae');
%     else
%         save('results.mat','mu_t','s_t','rmseErr','rmseErrn','rmstErr','rmstErrn','perceErr','perceErrn','perctErr','perctErrn','elapsedTime','params','prn','Marr');
%     end
%     allvars = whos;
%     tosave = cellfun(@isempty, regexp({allvars.class}, '^matlab\.(ui|graphics)\.'));
%     save('workspace.mat', allvars(tosave).name)
    for i = 1:2
        printres(i,:) = [rmseErr(i) rmseErrn(i) rmstErr(i) rmstErrn(i) perceErr(i) perceErrn(i) perctErr(i) perctErrn(i) elapsedTime(i) params(i)];
    end
else
    if system == 4
        save('results.mat','mu_t','s_t','rmseErr','rmstErr','perceErr','perctErr','elapsedTime','params','Marr');
        for i = 1:2
            printres(i,:) = [rmseErr(i)*1000 rmstErr(i)*1000 perceErr(i) perctErr(i) elapsedTime(i) params(i)];
        end
    else
        save('results.mat','mu_t','s_t','rmseErr','rmseErrn','rmstErr','rmstErrn','perceErr','perceErrn','perctErr','perctErrn','elapsedTime','params','Marr','Errae');
        for i = 1:4
            printres(i,:) = [rmseErr(i) rmseErrn(i) rmstErr(i) rmstErrn(i) perceErr(i) perceErrn(i) perctErr(i) perctErrn(i) elapsedTime(i) params(i)];
        end
    end
end

diary off
