function [x res] = omp_tensor(b,A,k,A_ip,normVec,tensDims,R)
% Based on code in SLIDES-Gribonval_Sparse_Methods_PartIV.pdf in Mestrado 
% folder.
% Adaptation of basic OMP (i.e. without Cholesky update for linear system)
% for HO-SuKro structured dictionary (sum of kronecker products). 
% It replaces the correlation calculation (A'*res) by a tensor mode-product.

I = length(tensDims);

[n,m] = size(A); 
x = zeros(m,1); 
res = b;
support = []; % empty support
for i=1:k 
% compute correlation between residual and columns of A
% corr = A.'*res;

corr = tmprod(reshape(res,tensDims),cellfun(@transpose, A_ip(:,1),'UniformOutput',false),1:I);
% corr = modeprod3(reshape(res,tensDims),A_ip{1,1}.',A_ip{2,1}.',A_ip{3,1}.',tensDims);
for p=2:R
    corr = corr + tmprod(reshape(res,tensDims),cellfun(@transpose, A_ip(:,p),'UniformOutput',false),1:I); 
%     corr = corr + modeprod3(reshape(res,tensDims),A_ip{1,p}.',A_ip{2,p}.',A_ip{3,p}.',tensDims);
end
corr = corr(:)./normVec;

% find position n (and value c) of the maximally correlated column 
[~, idx] = max(abs(corr));
% extend the support
support(end+1) = idx;
% update the representation 
x(support) = pinv(A(:,support))*b;
% update the residual
res  = b - A(:,support)*x(support);  
end