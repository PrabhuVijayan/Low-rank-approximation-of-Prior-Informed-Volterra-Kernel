%% Initial estimates of Wiener-PNLSS parameters from Volterra kernel tensor

if ~ismember(system,3:4)
    Ha = hankel(imp(3:end,1),[imp(end,1) zeros(1,Memory-1)]);
    delta = eye(Memory);
    gamma = Ha;
    C = gamma(1,:);
    B = delta(:,1);
    A = delta(:,2:end)/delta(:,1:end-1 );
    D = imp(2);

else
    ch(1) = H_kernel(1); p1 = 1;
    p2 = H_kernel(M+2)/ch(1);
    for i = 1:MemoryW+1
        if i == 1
            cw(i) = sqrt(H_kernel(M+1+i)/(p2*ch(1)));
        else
            cw(i) = H_kernel(M+1+i)/(2*p2*ch(1));
        end
    end

    normpar1 = cw./norm(cw);
    % using pinv and main diagonal
    v = [p2*(cw').^2; zeros(MemoryH,1)]';
    Tp = toeplitz([v(1) fliplr(v(MemoryW+2:end))], v)';
    Hktmp = zeros(M+1,1); val = 0; ccounter = 0;
    for i = 1:M+1
        Hktmp(i) = H_kernel((i-1)*(MemoryW+1)+M+2-val);
        if i >= MemoryH+2
            ccounter = ccounter + 1;
            val = val + ccounter;
        end
    end
    ch = (pinv(Tp)*Hktmp)';

    normpar2 = ch./norm(ch);

    v = [cw'; zeros(MemoryH,1)]';
    Tp = toeplitz([v(1) fliplr(v(MemoryW+2:end))], v)';
    p1 = (Tp*ch')'*H_kernel(1:M+1)/((Tp*ch')'*(Tp*ch'));
    
    S = cw'*cw;
    H2mat = zeros(Memory+1);
    for i = 1:MemoryH+1
        H2mat(i:i+MemoryW,i:i+MemoryW) = H2mat(i:i+MemoryW,i:i+MemoryW) + S.*ch(i);
    end
    H2matapp = zeros(Memory+1); ccounter = M+1; val = 0;
    for i = 1:Memory+1
        H2matapp(i,i:i+MemoryW-val) = H_kernel(ccounter+1:ccounter+1+MemoryW-val);
        ccounter = ccounter+1+MemoryW-val;
        if i >= MemoryW+1
            val = val+1;
        end
    end
    H2matapp = (H2matapp+H2matapp')/2;
    p2 = sum(H2matapp.*H2mat)/sum(H2mat.^2);

    S = reshape(cw,[length(cw),1,1]).*reshape(cw, [1,length(cw),1]).*reshape(cw,[1,1,length(cw)]);
    H2mat = zeros(Memory+1,Memory+1,Memory+1);
    for i = 1:MemoryH+1
        H2mat(i:i+MemoryW,i:i+MemoryW,i:i+MemoryW) = H2mat(i:i+MemoryW,i:i+MemoryW,i:i+MemoryW) + S.*ch(i);
    end
    H2matapp = zeros(Memory+1,Memory+1,Memory+1);
    o3 = fCombinations(M+NInputs,D1); o2 = find(U(1,:)==U(1,1)^3);
    for i = 1:length(o3)
        [~,col,val] = find(o3(i,:));
        if range(col) <= MemoryW
            sol = arrayfun(@(c, v) repmat(c, 1, v), col, val, 'UniformOutput', false);
            sol = [sol{:}]; % Concatenate into a single vector
            H2matapp(sol(1),sol(2),sol(3)) = H_kernel(o2);
            o2 = o2+1;
        end
    end
    indices = perms([1:length(size(H2matapp))]); e = zeros(size(H2matapp));
    for i = 1:size(indices,1)
       e = e + permute(H2matapp,indices(i,:)); 
    end
    H2matapp = e./size(indices,1); % division include after revision
    p3 = sum(H2matapp.*H2mat,"all")/sum(H2mat.^2,"all");

    pnl = [p1 p2 p3];
    
    D = p1*cw(1)*ch(1);
    C = [ch(1).*cw(2:end) ch(2:end)];
    B = [1; zeros(MemoryW-1,1); 1; zeros(MemoryH-1,1)];
    A = [compan([1 zeros(1,MemoryW)]) zeros(MemoryW,MemoryH); [cw(2:end); zeros(MemoryH-1,MemoryW)] compan([1 zeros(1,MemoryH)])];
end

switch system
    case {1,5,6,7}
        E = [];
        if ismember(system,5:7)
            F = H_kernel(M+3:end)';
        elseif system == 1 && simulation == 0
            F = zeros(NOutputs,size(fCombinations(M+NInputs,2:D1),1));
            pos = 1; start = 2;
            for ord = start:D1
                idx = generate_index(start:I,ord);
                for c = 1:size(idx,1)
                    indices = num2cell(unique(perms([idx(c,:),ones(1,Order-ord)]),'rows'));
                    [F,pos] = calcF(indices,F,pos,H);
                end
            end
            H_kernel = [D C F]';
        else
            F = H_kernel(M+2:end)';
        end

        % Method 3
        rank1mat = zeros(MemoryV+1); ccounter1 = 0;
        for i = 1:M+1
            ccounter = i*MemoryV - i*(i-3)/2;
            rank1mat(i,:) = [zeros(1,i-1) H_kernel(MemoryV+1+ccounter1+1:MemoryV+1+ccounter)'];
            ccounter1 = ccounter;            
        end
        rank1mat = (rank1mat+rank1mat')/2;
        [u1,s1,v1] = svd(rank1mat,'econ');
        cw = u1(:,1)';
        p2 = s1(1,1)*v1(:,1)'*u1(:,1);
        if simulation == 1
            figure, plot(coeff), hold on, plot(ncoeff), plot(cw), plot(normpar)
        end
        p1 = cw*H_kernel(1:M+1);
        cw3 = reshape(cw,[length(cw),1,1]).*reshape(cw, [1,length(cw),1]).*reshape(cw,[1,1,length(cw)]);
        Htmp = H(2:end,2:end,2:end);
        p3 = (cw3(:)'*Htmp(:))/(cw3(:)'*cw3(:));
        pnl = [p1 p2 p3];

    case 2
        vker = H_kernel(1:M+1)';%[H_kernel(M+1); H_kernel(1:M)]';
        normpar = vker./norm(vker);
        vker = normpar;
        nm = fCombinations(Memory+NInputs,true_model.nx);
        E = zeros(Memory,size(nm,1)); 
        F = zeros(NOutputs,size(nm,1));
        fpos = zeros(length(true_model.ny),1); start = 2;
        for i = 1:length(true_model.ny)
            temp = find(nm(:,end) == true_model.ny(i));
            fpos(i) = temp(1);
        end
        for ord = start:D1
            idx = generate_index(start:I,ord);
            idx(1:end-1,:) = [];
            for c = 1:size(idx,1)
                indices = num2cell(unique(perms([idx(c,:),ones(1,Order-ord)]),'rows'));
                [F,~] = calcF(indices,F,fpos(ord-1),H);
            end
        end
        for ord = start:D1
            idx = generate_index(start:I,ord);
            idx(find(range(idx,2) ~= 0),:) = [];
            idx(end,:) = [];
            Hvec = zeros(size(idx,1),1);
            for c = 1:size(idx,1)
                indices = num2cell(unique(perms([idx(c,:),ones(1,Order-ord)]),'rows'));
                [Hvec,~] = calcF(indices,Hvec,c,H);
            end
            E(:,fpos(ord-1)) = pinv(gamma)*Hvec; % E(2:end,fpos(ord-1)) = zeros(M-1,1);
        end
    case {3,4}
        xpowers = fCombinations(Memory+NInputs,true_model.nx);
        ypowers = fCombinations(Memory+NInputs,true_model.ny);
        F = ones(NOutputs,size(ypowers,1));
        E = [zeros(MemoryW,size(xpowers,1)); ones(MemoryH,size(xpowers,1))];
        temp = [cw(2:end) zeros(size(ch(2:end))) cw(1)];
        for i = 1:size(ypowers,1)
            for j = 1:length(temp)
                F(i) = F(i)*temp(j)^ypowers(i,j);
                E(MemoryW+1:end,i) = E(MemoryW+1:end,i)*temp(j)^xpowers(i,j);
            end
            if sum(ypowers(i,:)~=0) > 1
                F(i) = F(i)*sum(ypowers(i,:));
                E(MemoryW+1:end,i) = E(MemoryW+1:end,i)*sum(xpowers(i,:));
            end
            F(i) = F(i)*pnl(sum(ypowers(i,:)))*ch(1);
            E(MemoryW+1:end,i) = pnl(sum(ypowers(i,:))).*[1; zeros(MemoryH-1,1)].*E(MemoryW+1:end,i);
        end
    otherwise
        disp('other value')
end

%% Validation dataset

if system == 2
    Uv = [];
    for i = 1:D1
        Uv = [Uv toeplitz(uval(:).^i,[uval(1,1,1)^i zeros(1,M)])];
    end
elseif system == 3 || system == 4
%     Uv = toeplitz(uval(:),[uval(1,1,1) zeros(1,M)]);
%     o2 = fCombinations(M+NInputs,2:D1);
%     tmp = size(fCombinations(M+NInputs,2),1);
%     count = 0; cnt2 = 0;
%     for i = 1:size(o2,1)
%         count = count + 1;
%         [~,col,val] = find(o2(i,:));
%         rtmp = range(col);
%         if rtmp <= MemoryW
%             Uv(:,M+NInputs+i) = ones(size(uval(:)));
%             for j = 1:length(col)
%                 Uv(:,M+NInputs+i) = Uv(:,M+NInputs+i).*Uv(:,col(j)).^val(j);
%             end
%         elseif count == M+1-cnt2
%             count = 0; cnt2 = cnt2 + 1;
%         elseif count == MemoryW+1 && cnt2 == MemoryH
%             count = 0;
%         end
%         if i == tmp
%             count = 0; cnt2 = 0; cnt3 = 0;
%         end
%     end
%     Uv(:,all(Uv == 0))=[];
elseif system == 1
    o2 = fCombinations(M+NInputs,2:D1); 
    Uv = toeplitz(uval(:),[uval(1,1,1) zeros(1,M)]);
%     Uv = [Uv(:,2:end) Uv(:,1)];
    for i = 1:size(o2,1)
        [~,col,val] = find(o2(i,:));
        Uv(:,M+NInputs+i) = ones(size(uval(:)));
        for j = 1:length(col)
            Uv(:,M+NInputs+i) = Uv(:,M+NInputs+i).*Uv(:,col(j)).^val(j);
        end
    end
    if simulation == 0
        U = toeplitz(uest(:),[uest(1,1,1) zeros(1,M)]);
%         U = [U(:,2:end) U(:,1)];
        for i = 1:size(o2,1)
            [~,col,val] = find(o2(i,:));
            U(:,M+NInputs+i) = ones(size(uest(:)));
            for j = 1:length(col)
                U(:,M+NInputs+i) = U(:,M+NInputs+i).*U(:,col(j)).^val(j);
            end
        end
    end
elseif system == 5 || system == 6 || system == 7
    o2 = fCombinations(M+NInputs,2:D1); 
    Uv = toeplitz(uval(:),[uval(1,1,1) zeros(1,M)]);
%     Uv = [Uv(:,2:end) Uv(:,1)];
    for i = 1:size(o2,1)
        [~,col,val] = find(o2(i,:));
        Uv(:,M+NInputs+i) = ones(size(uval(:)));
        for j = 1:length(col)
            Uv(:,M+NInputs+i) = Uv(:,M+NInputs+i).*Uv(:,col(j)).^val(j);
        end
    end
%     Uv = [ones(length(uval(:)),1) Uv];
end

function [F,pos] = calcF(indices,F,pos,H)
    for l = 1:size(indices,1)
        F(pos) = F(pos) + H(indices{l,:});
    end
    pos = pos + 1;
end