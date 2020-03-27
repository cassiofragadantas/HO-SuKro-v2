function S = modeprod3_sparse(S,U1,U2,U3,size_tens,idx_12,idx_23,idx_31)
%MODEPROD3_SPARSE : This is a variation of MODEPROD3 which takes as an
%   input the permutation indexes (pre-calculated) to try and accelerate
%   the permutation operations related to the required unfolding
%   operations.
%
%   Conclusions: we cannot be faster than permute. The advantage is that
%   now we can manipulate sparse data (since we never pass through a tensor 
%   form), but this does not compensate in terms of execution time. The
%   reason is that only the first modeproduct uses sparsity.
%   
%   Related: see description of modeprod.m
%
%   Authors: Cassio Fraga Dantas (cassiofragadantas@gmail.com)

% size_tens = size(T);
N = length(size_tens);

% Cycle through the n-mode products.
%% Mode 1
% S = reshape(U1*reshape(S,size(S,1),[]),size_tens);
S = U1*S;
size_tens(1) = size(U1,1);

%% Mode 2
% From mode1 to mode2 unfold
if nargin < 8
    S_aux = zeros(size_tens(2),prod(size_tens([1 3:end])));
    for k = 1:prod(size_tens(3:end))
        S_aux(:,(k-1)*size_tens(1) + (1:size_tens(1))) = S(:,(k-1)*size_tens(2) + (1:size_tens(2))).';
    end
    S = reshape(S_aux,size_tens(2),[]);
else
    % Or simply, if idx_12 is available
    S = reshape(S(idx_12),size_tens(2),[]);
end
% size_tens = size_tens([2:N 1]);
size_tens(2) = size(U2,1);

% S = reshape(U2*reshape(S,size(S,1),[]),size_tens);
S = U2*S;

%% Mode 3
% From mode2 to mode3 unfold
if nargin < 8
    S_aux = zeros(size_tens(3),prod(size_tens([1:2 4:end])));
    for k = 1:size_tens(3)
        for kk = 1:prod(size_tens(4:end))
            S_aux(k,(kk-1)*prod(size_tens(1:2)) + (1:prod(size_tens(1:2)))) = reshape(S(:,(kk-1)*size_tens(1)*size_tens(3) + (k-1)*size_tens(1) +(1:size_tens(1))).',1,[]);
        end
    end
    S = reshape(S_aux,size_tens(3),[]);
else
    % Or simply, if idx_23 is available
    S = reshape(S(idx_23),size_tens(3),[]);
end
% size_tens = size_tens([2:N 1]);
size_tens(3) = size(U3,1);

S = U3*S;


%% Back to mode 1
if nargin < 8
    S_aux = zeros(size_tens(1),prod(size_tens(2:end)));
    for k = 1:prod(size_tens(4:end))
        S_aux(:,(k-1)*size_tens(2)*size_tens(3) + (1:size_tens(2)*size_tens(3))) = reshape(S(:,(k-1)*size_tens(1)*size_tens(2)+(1:size_tens(1)*size_tens(2))).',size_tens(1),[]);
    end
    S = reshape(S_aux,size_tens(1),[]);
else
    % Or simply, if idx_31 is available
    S = reshape(S(idx_31),size_tens(1),[]);
end
