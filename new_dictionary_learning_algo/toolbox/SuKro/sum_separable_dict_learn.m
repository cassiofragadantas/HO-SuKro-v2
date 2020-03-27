function [D,D_structured,X] = sum_separable_dict_learn(params)
%SuKro (Sum of Kroneckers) dictionary training.
%  [D,D_structured,X] = sum_separable_dict_learn(params) runs the dictionary 
%  training algorithm on the specified set of signals (as a field of the 
%  input variable 'params'), returning the trained dictionary 'D', its
%  version before column normalization 'D_structured'
%  and the sparse signal representation matrix 'X'.
%
%  The training algorithm performs an alternate minimization on the
%  variables 'D' (dictionary update step) and 'X' (sparse coding step).
%
%  1) Dictionary update step
%     Considers the following optimization problem
%
%         min  |Y-D*X|_F^2 + lambda*rank( R(D) )
%         D
%
%     where the R() is the rearrangement operator (see [3] for details).
%
%  2) Sparse coding step
%     Two modes of operation for the sparse reconstruction step are
%     available:
%
%     - Sparsity-based minimization, the optimization problem is given by
%
%         min  |Y-D*X|_F^2      s.t.  |X_i|_0 <= T
%         D,X
%
%       where Y is the set of training signals, X_i is the i-th column of
%       X, and T is the target sparsity. 
%  
%     - Error-based minimization, the optimization problem is given by
%
%         min  |X|_0      s.t.  |Y_i - D*X_i|_2 <= EPSILON
%         D,X
%
%       where Y_i is the i-th training signal, EPSILON is the target error.
%
%  --------------------------
%
%  The input arguments organization in this code is based on the K-SVD code
%  available at:  http://www.cs.technion.ac.il/~ronrubin/software.html
%  References:
%  [1] M. Aharon, M. Elad, and A.M. Bruckstein, "The K-SVD: An Algorithm
%      for Designing of Overcomplete Dictionaries for Sparse
%      Representation", the IEEE Trans. On Signal Processing, Vol. 54, no.
%      11, pp. 4311-4322, November 2006.
%  [2] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation
%      of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit",
%      Technical Report - CS, Technion, April 2008.
%
%  Required fields in PARAMS:
%  --------------------------
%
%    'data' - Training data.
%      A matrix containing the training signals as its columns.
%
%    'Tdata' / 'Edata' - Sparse coding target.
%      Specifies the number of coefficients (Tdata) or the target error in
%      L2-norm (Edata) for coding each signal. If only one is present, that
%      value is used. If both are present, Tdata is used, unless the field
%      'codemode' is specified (below).
%
%    'initdict' - Specifies the initial dictionary for the training. It
%      should be a matrix of size nxm, where n=size(data,1).
%
%
%  Optional fields in PARAMS:
%  --------------------------
%
%    'iternum' - Number of training iterations.
%      If not specified, the default is 10.
%
%    'alpha' - Controls the penalization on the displacement rank.
%      It determines the regularization parameter lambda as follows:
%      
%        lambda = params.alpha*norm(Y);
%
%    'memusage' - Memory usage.
%      This parameter controls memory usage of the function. 'memusage'
%      should be one of the strings 'high', 'normal' (default) or 'low'.
%      When set to 'high', the fastest implementation of OMP is used, which
%      involves precomputing both G=D'*D and DtX=D'*X. This increasese
%      speed but also requires a significant amount of memory. When set to
%      'normal', only the matrix G is precomputed, which requires much less
%      memory but slightly decreases performance. Finally, when set to
%      'low', neither matrix is precomputed. This should only be used when
%      the trained dictionary is highly redundant and memory resources are
%      very low, as this will dramatically increase runtime. See function
%      OMP for more details.
%
%    'codemode' - Sparse-coding target mode.
%      Specifies whether the 'Tdata' or 'Edata' fields should be used for
%      the sparse-coding stopping criterion. This is useful when both
%      fields are present in PARAMS. 'codemode' should be one of the
%      strings 'sparsity' or 'error'. If it is not present, and both fields
%      are specified, sparsity-based coding takes place.
%
%
%  Optional fields in PARAMS - advanced:
%  -------------------------------------
%
%    'maxatoms' - Maximal number of atoms in signal representation.
%      When error-based sparse coding is used, this parameter can be used
%      to specify a hard limit on the number of atoms in each signal
%      representation (see parameter 'maxatoms' in OMP2 for more details).
%
%
%   Summary of all fields in PARAMS:
%   --------------------------------
%
%   Required:
%     'data'                   training data
%     'Tdata' / 'Edata'        sparse-coding target
%     'initdict'               initial dictionary / dictionary size
%
%   Optional (default values in parentheses):
%     'iternum'                number of training iterations (10)
%     'u'                      ADMM \mu coefficient (1e7)
%     'memusage'               'low, 'normal' or 'high' ('normal')
%     'codemode'               'sparsity' or 'error' ('sparsity')
%     'maxatoms'               max # of atoms in error sparse-coding (none)
%     'kro_dims'               dimensions of the subdictionaries 
%                              (N1=N2=sqrt(N), M1=M2=sqrt(M))
%
%
%  Reference:
%  [3] C.F. Dantas, M. N. da Costa, R.R. Lopes, "Learning Dictionaries as a
%      sum of Kronecker products", To appear.



%% Required parameters

if (isfield(params,'initdict'))
    D = params.initdict;
else
    error('Initial dictionary should be provided in field params.initdict');
end

if (isfield(params,'data'))
    Y = params.data;
else
    error('Training data should be provided in field params.data');
end


if (size(Y,2) < size(D,2))
  error('Number of training signals is smaller than number of atoms to train');
end

% subdictionaries dimensions
if (isfield(params,'kro_dims'))
    N1 = params.kro_dims.N1;
    N2 = params.kro_dims.N2;
    M1 = params.kro_dims.M1;
    M2 = params.kro_dims.M2;    
else
    N1 = sqrt(size(D,1));
    N2 = sqrt(size(D,1));
    M1 = sqrt(size(D,2));
    M2 = sqrt(size(D,2));
end

assert( N1*N2==size(D,1) && M1*M2==size(D,2), ...
        'N (resp. M) should be equal to N1*N2 (resp. M1*M2)')
assert( round(N1)==N1 && round(N2)==N2 && round(M1)==M1 && round(M2)==M2,...
        'N1,N2,M1 and M2 should all be integers')

%% Parameter setting

% iteration count %

if (isfield(params,'iternum'))
  iternum = params.iternum;
else
  iternum = 10;
end

% ADMM coefficient
if (isfield(params,'u'))
  u = params.u;
else
  u = 1e7; % possible improvement: automatically set a good u.
end

idx_vec = 1:N1*N2*M1*M2;
[idx] = reord(reshape(idx_vec,N1*N2, M1*M2),N1,N2,M1,M2);
[idx_inv] = reord_inv(reshape(idx_vec,N1*M1, N2*M2),N1,N2,M1,M2);
idx = idx(:);
idx_inv = idx_inv(:);
clear idx_vec


%% Sparse Coding parameter setting
CODE_SPARSITY = 1;
CODE_ERROR = 2;

MEM_LOW = 1;
MEM_NORMAL = 2;
MEM_HIGH = 3;

%%%%% parse input parameters %%%%%

ompparams = {'checkdict','off'};

% coding mode %

if (isfield(params,'codemode'))
  switch lower(params.codemode)
    case 'sparsity'
      codemode = CODE_SPARSITY;
      thresh = params.Tdata;
    case 'error'
      codemode = CODE_ERROR;
      thresh = params.Edata;
    otherwise
      error('Invalid coding mode specified');
  end
elseif (isfield(params,'Tdata'))
  codemode = CODE_SPARSITY;
  thresh = params.Tdata;
elseif (isfield(params,'Edata'))
  codemode = CODE_ERROR;
  thresh = params.Edata;

else
  error('Data sparse-coding target not specified');
end


% max number of atoms %

if (codemode==CODE_ERROR && isfield(params,'maxatoms'))
  ompparams{end+1} = 'maxatoms';
  ompparams{end+1} = params.maxatoms;
end


% memory usage %

if (isfield(params,'memusage'))
  switch lower(params.memusage)
    case 'low'
      memusage = MEM_LOW;
    case 'normal'
      memusage = MEM_NORMAL;
    case 'high'
      memusage = MEM_HIGH;
    otherwise
      error('Invalid memory usage mode');
  end
else
  memusage = MEM_NORMAL;
end

% omp function %

if (codemode == CODE_SPARSITY)
  ompfunc = @omp;
else
  ompfunc = @omp2;
end


% data norms %

XtX = []; XtXg = [];
if (codemode==CODE_ERROR && memusage==MEM_HIGH)
  XtX = sum(Y.^2);
end

err = zeros(1,iternum);
gerr = zeros(1,iternum);

if (codemode == CODE_SPARSITY)
  errstr = 'RMSE';
else
  errstr = 'mean atomnum';
end

%% Sparse Coding
G = [];
if (memusage >= MEM_NORMAL)
    G = D'*D;
end

if (memusage < MEM_HIGH)
  X = ompfunc(D,Y,G,thresh,ompparams{:});

else  % memusage is high

  if (codemode == CODE_SPARSITY)
    X = ompfunc(D'*Y,G,thresh,ompparams{:});

  else
    X = ompfunc(D'*Y,XtX,G,thresh,ompparams{:});
  end

end

%%%%%%%%%%%%%%%%%%

%% Optimization %%
%%%%%%%%%%%%%%%%%%
lambda = params.alpha*norm(Y);

%% Ajuste automatico para step size
step = 1e-10/norm(D);

found = false;

iter = 0;
clear lbound rbound
while ~found
    Z = reord(D,N1,N2,M1,M2,idx)/u;
    D_hat = zeros(size(D));
    
    % Min Dmod
    [U, S, V] = svd(reord(D_hat,N1,N2,M1,M2,idx) + Z ,'econ');
    diagS = diag(S);
    svp = length(find(diagS > lambda/u));
    temp_Dmod = U(:, 1:svp) * diag(diagS(1:svp) - lambda/u) * V(:, 1:svp)';
    % Min D_hat
    for k = 1:12
        grad1 = D_hat - reord_inv(temp_Dmod - Z,N1,N2,M1,M2,idx_inv);
        grad2 = (D_hat*X - Y)*X.';
        if k == 10
            norm1 = norm(step*(u*grad1 + grad2), 'fro');
        end
        D_hat = D_hat - step*(u*grad1 + grad2);
    end
    norm2 = norm(step*(u*grad1 + grad2), 'fro');

    if norm2 < norm1 % Converges
       lbound = step;
       if exist('rbound','var')
           step = (lbound + rbound)/2;
       else
           step = step*1e3;
       end
    else % Diverges
        rbound = step;
        if exist('lbound','var')
           step = (lbound + rbound)/2;
        else
           step = step/1e3;
        end
    end

    if exist('lbound','var') && exist('rbound','var') 
        if (rbound - lbound)/lbound < 1e-6
            found = true;
            step = step/4;
        end
    end
    iter = iter + 1;
end
fprintf('Step-size: %2.2E\n',step)

%% Alternating Optimization
success = false;
fprintf('Iteration:              ')
while ~success
    try
        Z = reord(D,N1,N2,M1,M2,idx)/u;  % It is faster if  these variables are not reseted every iteration
        D_hat = zeros(size(D));
        tic
        for k = 1:iternum
            fprintf(repmat('\b',1,13));
            fprintf('%4d / %4d  ',k,iternum);
            
            %% Dictionary update step
            tol = 1e-4*norm(D,'fro');
            if k == iternum, tol = 1e-7*norm(D,'fro'); end % Better accuracy on the last iteration
            converged = false;
            iter = 1;

            % ADMM Loop
            while ~converged
                % Min Dmod
                [U, S, V] = svd(reord(D_hat,N1,N2,M1,M2,idx) + Z ,'econ');
                diagS = diag(S);
                svp = length(find(diagS > lambda/u));
                temp_Dmod = U(:, 1:svp) * diag(diagS(1:svp) - lambda/u) * V(:, 1:svp)';
                % Min D_hat
                grad1 = D_hat - reord_inv(temp_Dmod - Z,N1,N2,M1,M2,idx_inv);
                grad2 = (D_hat*X - Y)*X.';
                temp_D_hat = D_hat - step*(u*grad1 + grad2);
                D_hat = temp_D_hat;

                Dmod = temp_Dmod;


                temp_Z = Z - Dmod + reord(D_hat,N1,N2,M1,M2,idx);

                % stop Criterion    
                if norm(temp_Z - Z, 'fro') < tol
                    converged = true;
                    % disp(['Total nº of iterations: ' num2str(iter)]);
                end

                Z = temp_Z;

                iter = iter+1;
                if iter > 1e6, error('ALM did not converge, probably due to bad parameter setting. Retrying...'),end
            end
            % Dict_improvement = norm(normc(D_hat) - D)
            D = normc(D_hat);
            
            %% Sparse coding step
            G = [];
            if (memusage >= MEM_NORMAL)
                G = D'*D;
            end

            if (memusage < MEM_HIGH)
              X = ompfunc(D,Y,G,thresh,ompparams{:});

            else  % memusage is high

              if (codemode == CODE_SPARSITY)
                X = ompfunc(D'*Y,G,thresh,ompparams{:});

              else
                X = ompfunc(D'*Y,XtX,G,thresh,ompparams{:});
              end

            end     
        end
        total_time = toc; fprintf('    Elapsed time: %.1fs\n',total_time);
        success = true;
    catch err
        disp(err.message); disp('Trying smaller step.')
        step = step/2
    end 
end

D_structured = D_hat;
end
