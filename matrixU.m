function U = matrixU(u,D,M,I,ker)
    [N,P] = size(u);
    U = zeros(N,I);
    u = [zeros(M,P); u];
    for i = M:N+M-1            
	    temp = ones(1,I);
        for j = 1:M+1
            temp(2+(j-1)*P:2+j*P-1) = u(2+i-j,:);                
        end
        U(i-M+1,:) = kronProd(temp,D);
    end
end