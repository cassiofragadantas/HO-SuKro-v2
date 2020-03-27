function [sols, y_est, iters, activationHist] = SolveOMP_tensor_only(y, solDims, A_ip,normVec,tensDims,R, maxIters, lambdaStop, solFreq, verbose, OptTol)
%SolveOMP_tensor: based on SolveOMP function from SparseLab toolbox. It
%   replaces the correlation calculation (A'*res) by a tensor mode-product,
%   supposing the dictionary A to have a HO-SuKro structure (i.e. a sum of
%   Kronecker products).
%
%   Author: Cassio Fraga Dantas
% 
% SolveOMP: Orthogonal Matching Pursuit
% Usage
%	[sols, iters, activationHist] = SolveOMP(A, y, N, maxIters, lambdaStop, solFreq, verbose, OptTol)
% Input
%	y           vector of length n.
%   N           length of solution vector. 
%   A_ip        Cell array containing the  IxR kronecker terms forming the
%               dictionary matrix A, such that: 
%               A = \sum_{p=1}^R kron(A_1p, ... AIp)
%	maxIters    maximum number of iterations to perform. If not
%               specified, runs to stopping condition (default)
%   lambdaStop  If specified, the algorithm stops when the last coefficient 
%               entered has residual correlation <= lambdaStop. 
%   solFreq     if =0 returns only the final solution, if >0, returns an 
%               array of solutions, one every solFreq iterations (default 0). 
%   verbose     1 to print out detailed progress at each iteration, 0 for
%               no output (default)
%	OptTol      Error tolerance, default 1e-5
% Outputs
%	 sols            solution(s) of OMP
%	 y_est           D*sols, the reconstructed data y
%    iters           number of iterations performed
%    activationHist  Array of indices showing elements entering  
%                    the solution set
% Description
%   SolveOMP is a greedy algorithm to estimate the solution 
%   of the sparse approximation problem
%      min ||x||_0 s.t. A*x = b
%   The implementation implicitly factors the active set matrix A(:,I)
%   using Cholesky updates. 
%
%   The dictionary matrix A is not provided directly, but rather by means
%   its composing Kroneckers terms in the cell array A_ip.
%
%   This function gets as input a vector x and an index set I, and returns
%   y = A(:,I)*x if mode = 1, or y = A(:,I)'*x if mode = 2. 
%   A is the m by dim implicit matrix implemented by the function. I is a
%   subset of the columns of A, i.e. a subset of 1:dim of length n. x is a
%   vector of length n is mode = 1, or a vector of length m is mode = 2.
% See Also
%   SolveLasso, SolveBP, SolveStOMP
%

if nargin < 11,
	OptTol = 1e-5;
end
if nargin < 10,
    verbose = 0;
end
if nargin < 9,
    solFreq = 0;
end
if nargin < 8,
    lambdaStop = 0;
end
if nargin < 7,
    maxIters = length(y);
end

n = length(y);
I = length(tensDims);
N = prod(solDims);

% Parameters for linsolve function
% Global variables for linsolve function
% global opts opts_tr machPrec
opts.UT = true; 
opts_tr.UT = true; opts_tr.TRANSA = true;
machPrec = 1e-5;

% Initialize
x = zeros(N,1);
k = 0;
R_I = [];
activeSet = [];
A_activeSet = [];
sols = [];
res = y;
normy = norm(y);
resnorm = normy;
done = 0;
% Testing norm of residual before starting. Cassio: I added this.
if (resnorm <= OptTol)
    done = 1;
    sols = [sols x];
    y_est = y - res;
end

while ~done
    % Calculate correlation

    % OPTION 1: for loop implementation
%     corr = tmprod(reshape(res,tensDims),cellfun(@transpose, A_ip(:,1),'UniformOutput',false),1:I);
    corr = modeprod3(reshape(res,tensDims),A_ip{1,1}.',A_ip{2,1}.',A_ip{3,1}.',tensDims); % modeprod3 works only for I = 3
    for p=2:R
%         corr = corr + tmprod(reshape(res,tensDims),cellfun(@transpose, A_ip(:,p),'UniformOutput',false),1:I); 
        corr = corr + modeprod3(reshape(res,tensDims),A_ip{1,p}.',A_ip{2,p}.',A_ip{3,p}.',tensDims);
    end    
    % OPTION 2: no-for implementation (seems to be slower)
%     corr = cellfun(@(x,y,z) modeprod3(reshape(res,tensDims),x.',y.',z.',tensDims), A_ip(1,:),A_ip(2,:),A_ip(3,:), 'UniformOutput',false); % inner product between block columns
%     corr = sum(cat(4,corr{:}),4);
    
    corr = corr(:)./normVec;

    % Matching step
    [maxcorr, i] = max(abs(corr));
    newIndex = i(1);
    
    % Obtain new atom from blocks A_ip and append to current (restricted) dictionary
    [i3, i2, i1] = index_map(newIndex,solDims);  %TODO generalize index_map to arbitrary nb modes. Output a matrix with I rows.
    newVec = kron(A_ip{3,1}(:,i3),A_ip{2,1}(:,i2),A_ip{1,1}(:,i1));
    for p = 2:R
        newVec = newVec + kron(A_ip{3,p}(:,i3),A_ip{2,p}(:,i2),A_ip{1,p}(:,i1));
    end
    newVec = newVec/normVec(newIndex);
    
    % Update Cholesky factorization of A_I
    [R_I, ~] = updateChol(R_I, A_activeSet, newVec, activeSet, opts_tr, machPrec);
    activeSet = [activeSet newIndex];
    

    A_activeSet = [A_activeSet newVec];

    % Solve for the least squares update: (A_I'*A_I)dx_I = corr_I
    dx = zeros(N,1);
    z = linsolve(R_I,corr(activeSet),opts_tr);
    dx(activeSet) = linsolve(R_I,z,opts);
    x(activeSet) = x(activeSet) + dx(activeSet);

    % Compute new residual
    res = y - A_activeSet * x(activeSet); % If activeSet is big enough 
    resnorm = norm(res);

    %if ((resnorm <= OptTol*normy) | ((lambdaStop > 0) & (maxcorr <= lambdaStop)))
    if ((resnorm <= OptTol) | ((lambdaStop > 0) & (maxcorr <= lambdaStop)))
        done = 1;
    end

    if verbose
        fprintf('Iteration %d: Adding variable %d\n', k, newIndex);
    end

    k = k+1;
    if k >= maxIters
        done = 1;
    end

    if done | ((solFreq > 0) & (~mod(k,solFreq)))
        sols = [sols x];
        y_est = y - res;
    end
end

iters = k;
activationHist = activeSet;
% clear opts opts_tr machPrec


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [R, flag] = updateChol(R, A_activeSet, newVec, activeSet, opts_tr, machPrec)
% updateChol: Updates the Cholesky factor R of the matrix 
% A(:,activeSet)'*A(:,activeSet) by adding A(:,newIndex)
% If the candidate column is in the span of the existing 
% active set, R is not updated, and flag is set to 1.

% global opts_tr machPrec
flag = 0;

if isempty(activeSet),
    R = sqrt(sum(newVec.^2));
else
    p = linsolve(R,A_activeSet'*newVec,opts_tr); % This product (A_activeSet'*newVec) could use the structure. But it is not worth it.

    q = sum(newVec.^2) - sum(p.^2);
    if (q <= machPrec) % Collinear vector
        flag = 1;
    else
        R = [R p; zeros(1, size(R,2)) sqrt(q)];
    end
end

%
% Copyright (c) 2006. Yaakov Tsaig
%  

%
% Part of SparseLab Version:100
% Created Tuesday March 28, 2006
% This is Copyrighted Material
% For Copying permissions see COPYING.m
% Comments? e-mail sparselab@stanford.edu
%
