function out = generate_index(N,k)
    v = (1:numel(N)).';
    bl = true;
    for i = 1:k-1
        vtmp = reshape(v,[ones(1,i) numel(v)]);
        bl = bl & (v >= vtmp);
        v = vtmp;
    end
    c = cell(1,k);
    [c{:}] = ind2sub(repmat(numel(N),1,k),find(bl));
    idx = [c{:}];
    out = N(idx);
end