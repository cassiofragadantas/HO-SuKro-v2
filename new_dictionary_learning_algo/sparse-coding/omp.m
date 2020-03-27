function [x res] = omp(b,A,k)
% Based on code in SLIDES-Gribonval_Sparse_Methods_PartIV.pdf in Mestrado 
% folder.
% Basic OMP (i.e. without Cholesky update for linear system)

[n,m] = size(A); 
x = zeros(m,1); 
res = b;
support = []; % empty support
for i=1:k 
% compute correlation between residual and columns of A
corr = A.'*res;

% find position n (and value c) of the maximally correlated column 
[~, idx] = max(abs(corr));
% extend the support
support(end+1) = idx;
% update the representation 
x(support) = pinv(A(:,support))*b;
% update the residual
res  = b - A(:,support)*x(support);  
end