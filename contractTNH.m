function H = contractTNH(rTNH)
    n = 4;
    D = size(rTNH,2);
    s = ones(D,n);
    sb = zeros(1,D*(n-2));
    for i = 1:D
        sa = size(rTNH{i});
        sb((i-1)*(n-2)+1:i*(n-2)) = flip(sa(2:3)); 
        if i == 1 || i == D
            rTNH{i} = permute(rTNH{i},[4,3,2,1]);
        else
            rTNH{i} = permute(rTNH{i},[3,4,2,1]);
        end
        if numel(size(rTNH{i})) ~= n
            s(i,1:numel(size(rTNH{i}))) = size(rTNH{i});
        else
            s(i,:) = size(rTNH{i});
        end
    end
    H = reshape(rTNH{end},[prod(s(end,1:end-1)) s(end,end)]);
    for i = D-1:-1:1
        if i == 1
            temp = reshape(rTNH{i},[s(i,1) prod(s(i,2:end))]);
        else
            temp = reshape(rTNH{i},s(i,2),[]);
        end
        H = H*temp;
        H = reshape(H,[prod(sb((i-1)*(n-2)+1:end)) numel(H)/prod(sb((i-1)*(n-2)+1:end))]);
    end
    flipsb = zeros(1,D*(n-2));
    for i = 1:D
        flipsb((n-2)*(i-1)+1:(n-2)*i) = sb((n-2)*(D-i+1)-1:(n-2)*(D-i+1));
    end
    H = reshape(H,flipsb);
    I = zeros(1,D*(n-2));
    for i = 1:n-2
        I((i-1)*D+1:i*D) = i:n-2:D*(n-2);
    end
    H = permute(H,I);
    sb = sb(I);
    b = ones(1,n-2);
    for i = 1:n-2
        b(i) = prod(sb((i-1)*D+1:i*D));
    end
    if length(b) ~= 1
        H = reshape(H,b);
    end
end