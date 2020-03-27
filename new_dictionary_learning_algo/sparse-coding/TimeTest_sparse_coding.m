% Time tests for OMP sparse coding.
%
% Notation: 
% y: data vector ∈ \R^{prod(n)}
% D: dictionary ∈ \R^{prod(n) x prod(m)}
% x: sparse coefficient vector ∈ \R^{prod(n) x prod(m)}


% Parameters
n = [10 10 10];  % data dimensions
m = [20 20 20];  % nb. of atoms per sub-dictionary. prod(m) = nb. of atoms
nnz_x = 10;     % nb. of OMP iterations
R_vec = 1;      % vector containing all nb. of kronecker summing terms to be tested

mc_it = 5000;    % nb. of runs to be averaged 

for R = R_vec
    R
    % Generating dictionary
    I = length(n);      % nb. of modes
    D_ip = cell(I,R);   % HO-SuKro dictionary
    D = zeros(prod(n),prod(m));
    for p = 1:R 
        for i = 1:I
            D_ip{i,p} = randn(n(i),m(i)); % Random sub-dictionaries
            D_ip{i,p} = D_ip{i,p}*diag(1./sqrt(sum(D_ip{i,p}.^2))); % Normalize columns
        end
        D = D + kron(D_ip(I:-1:1,p));
    end
    normVec = sqrt(sum(D.^2));
    D = D*diag(1./normVec); % Normalize columns. mean(sqrt(sum(D.^2))) approx. sqrt(m)

    % Testing that products with dictionaries give same result
%     y = randn(prod(n),1); % random data
%     teste = D.'*y;
%     teste2 = zeros(m);
%     for p = 1:R
%         teste2 = teste2 + tmprod(reshape(y,n),cellfun(@transpose, D_ip(:,p),'UniformOutput',false),1:I);
%     end
%     teste2 = teste2(:)./normVec.';
%     assert(norm(teste-teste2(:),'fro')<eps*prod(m),'Structured product does not give same result as normal product')

    % Run OMP
    time_OMP = zeros(1,mc_it);                  % Naive OMP
    time_OMP_tensor = zeros(1,mc_it);           % Naive OMP with structured dict.
    time_OMP_Cholesky = zeros(1,mc_it);         % Cholesky-OMP
    time_OMP_Cholesky_tensor = zeros(1,mc_it);  % Cholesky-OMP with structured dict.
    time_OMP_Cholesky_tensor_only = zeros(1,mc_it);  % Cholesky-OMP with structured dict. - FULL DICTIONARY NOT REQUIRED

    for k = 1:mc_it
        y = randn(prod(n),1); % random data

        % Naive OMP
        tic, x_ref = omp(y,D,nnz_x);  time_OMP(k) = toc;
        % Naive OMP with structured dict.
        tic, x = omp_tensor(y,D,nnz_x,D_ip,normVec.',n,R);  time_OMP_tensor(k) = toc;
        assert(norm(x-x_ref,'fro')<eps*prod(m), 'OMP does not give same result')
        % Cholesky-OMP
        tic, x = SolveOMP(D, y, prod(m), nnz_x);  time_OMP_Cholesky(k) = toc;
        assert(norm(x-x_ref,'fro')<eps*prod(m), 'OMP does not give same result')
        % Cholesky-OMP with structured dict.
        tic, x = SolveOMP_tensor(D, y, prod(m), D_ip,normVec.',n,R, nnz_x);  time_OMP_Cholesky_tensor(k) = toc;
        assert(norm(x-x_ref,'fro')<eps*prod(m), 'OMP does not give same result')
        % Cholesky-OMP with structured dict. - FULL DICTIONARY NOT REQUIRED
        tic, x = SolveOMP_tensor_only(y, m, D_ip,normVec.',n,R, nnz_x);  time_OMP_Cholesky_tensor_only(k) = toc;
        assert(norm(x-x_ref,'fro')<eps*prod(m), 'OMP does not give same result')
    end
    save(strcat('timeResults_OMP_n',sprintf('%.0f_' , n), 'm', sprintf('%.0f_' , m), ...
                'R', num2str(R),'_nnz',num2str(nnz_x),'_it',num2str(mc_it)), ...
         'time_OMP','time_OMP_tensor','time_OMP_Cholesky','time_OMP_Cholesky_tensor','time_OMP_Cholesky_tensor_only'); 
end

%% PROFILE
% The order in which the profiles are made doesn't change results (was tested)

Y = randn(prod(n),mc_it);

profile on
for k = 1:mc_it
    x = omp(Y(:,k),D,nnz_x);
end
profile off
profsave(profile('info'), ...
         strcat('myprofile_OMP_naive_n',sprintf('%.0f_' , n), 'm', sprintf('%.0f_' , m), ...
                'nnz',num2str(nnz_x),'_it',num2str(mc_it)))

profile on
for k = 1:mc_it
    x = SolveOMP(D, Y(:,k), prod(m), nnz_x+1);
end
profile off
profsave(profile('info'), ...
         strcat('myprofile_OMP_Cholesky_n',sprintf('%.0f_' , n), 'm', sprintf('%.0f_' , m), ...
                'nnz',num2str(nnz_x),'_it',num2str(mc_it)))

G = D.'*D;            
profile on
for k = 1:mc_it
    x = SolveOMP_Gram(D, Y(:,k), prod(m), G, nnz_x+1);
end
profile off
profsave(profile('info'), ...
         strcat('myprofile_OMP_Cholesky_Gram_n',sprintf('%.0f_' , n), 'm', sprintf('%.0f_' , m), ...
                'nnz',num2str(nnz_x),'_it',num2str(mc_it)))
            
profile on
for k = 1:mc_it
    x = SolveOMP_tensor(D, Y(:,k), prod(m), D_ip,normVec.',n,R, nnz_x+1);
end
profile off
profsave(profile('info'), ...
         strcat('myprofile_OMP_Cholesky_tensor_R',num2str(R),'_n',sprintf('%.0f_' , n), 'm', sprintf('%.0f_' , m), ...
                'nnz',num2str(nnz_x),'_it',num2str(mc_it)))            