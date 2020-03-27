function [i, j, k] = index_map(l,dims)
% This function returns the triplet of indexes i,j,k corresponding to each
% matrix index in a Kronecker product for column l. The kronecker factors
% have column dimensions dims.

J = dims(2); K = dims(3);

k = mod(l-1,K)+1;
j = mod((l-k)/K,J)+1;
i = (l-k-K*(j-1))/J/K+1;

end