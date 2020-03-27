function S = modeprod3(S,U1,U2,U3,size_tens)
%MODEPROD Mode-n tensor-matrix product (my version of tmprod from tensorlab)
%   This is a homemade version of tmprod from tensorlab, removing as much
%   logical overhead as possible in order to obtain better performance.
%   We will suppose all input arguments to be correctly provided. Options
%   (like transpose) are removed.
%
%   S = modeprod(T,U,mode) computes the tensor-matrix product of the tensor
%   T with the matrices U{1}, ..., U{N} along the modes mode(1), ...,
%   mode(N), respectively. Note that in this implementation, the vector
%   mode should contain distinct integers. The mode-n tensor-matrix
%   products are computed sequentially in a heuristically determined order.
%   A mode-n tensor-matrix product results in a new tensor S in which the
%   mode-n vectors of a given tensor T are premultiplied by a given matrix
%   U{n}, i.e., tens2mat(S,mode(n)) = U{n}*tens2mat(T,mode(n)).
%
%   [S,iperm] = tmprod(T,U,mode) and S = tmprod(T,U,mode,'saveperm') save
%   one permutation operation. In the former case, the tensor-matrix
%   product can then be recovered by permute(S,iperm).
%
%   Authors: Cassio Fraga Dantas (cassiofragadantas@gmail.com)

% size_tens = size(T);
N = length(size_tens);

% Sort the order of the mode-n products.
% [~,idx] = sort(size_tens(mode)./cellfun('size',U,m(1)));
% mode = mode(idx);
% U = U(idx);

% Compute the complement of the set of modes.
% n = length(mode);
% N = length(size_tens);
% bits = ones(1,N);
% bits(mode) = 0;
% modec = 1:N;
% modec = modec(logical(bits(modec)));

% Prepermute the tensor.
% perm = [mode modec];
% size_tens = size_tens(perm);
% S = T; if any(mode ~= 1:n), S = permute(S,perm); end

% S = T;

% Cycle through the n-mode products.
size_tens(1) = size(U1,1);

S = reshape(U1*reshape(S,size(S,1),[]),size_tens);

%     if i < n
    S = permute(S,[2:N 1]);
    size_tens = size_tens([2:N 1]);
%     end

size_tens(1) = size(U2,1);

S = reshape(U2*reshape(S,size(S,1),[]),size_tens);

%     if i < n
    S = permute(S,[2:N 1]);
    size_tens = size_tens([2:N 1]);
%     end

size_tens(1) = size(U3,1);

S = reshape(U3*reshape(S,size(S,1),[]),size_tens);

%     if i < n
%     S = permute(S,[2:N 1]);
    S = permute(S,[N-1:N 1:N-2]);
%     size_tens = size_tens([2:N 1]);
%     end


% % Permutation and inverse permutation
% %perm = [m 1:m-1 m+1:ndims(T)];
% perm = randperm(ndims(T));
% T_perm = permute(T,perm);
% iperm(perm) = 1:ndims(T); %inverse permutation
% T2 = permute(T_perm,iperm); % It is the same as T
% max(T(:)-T2(:));
% 
% % From a permutation direcly to another, without inversing first
% c = 2;
% perm = [c 1:c-1 c+1:ndims(T)]; % perm before
% T_perm = permute(T,perm);
% c = 3;
% perm2 = [c 1:c-1 c+1:ndims(T)]; % aimed perm
% T_perm2 = permute(T,perm2);
% perm3(perm(perm2)) = 1:ndims(T);
% T_perm3 = permute(T_perm,perm3); % should be equal to T_perm2
% max(T_perm2(:)-T_perm3(:));





% Inverse permute the tensor, unless the user intends to do so himself.
% iperm(perm([n:N 1:n-1])) = 1:N;
% if nargout <= 1 && ~saveperm, S = permute(S,iperm); end

% S = permute(S,[N 1:N-1]);