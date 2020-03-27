% Time tests for tensorial mode-product, compared to unstructured
% matrix-vector operation.
%
% Notation: 
% y: data vector ∈ \R^{prod(n)}
% D: dictionary ∈ \R^{prod(n) x prod(m)}
% x: sparse coefficient vector ∈ \R^{prod(n) x prod(m)}


% Parameters
n = [5 5 5 ];      % data dimensions
m = [10 10 10];  % nb. of atoms per sub-dictionary. prod(m) = nb. of atoms
R_vec = 1:5;        % vector containing all nb. of kronecker summing terms to be tested

% mc_it (nb. of runs to be averaged ) is defined within the RC.m script
% sparse (use sparse data if sparse=true) defined within the RC.m script

for R = R_vec
    R
    cpd_rank = R;
    % Generating dictionary
    I = length(n);      % nb. of modes
    D_terms = cell(I,R);   % HO-SuKro dictionary
    D = zeros(prod(n),prod(m));
    for p = 1:R 
        for i = 1:I
            D_terms{i,p} = randn(n(i),m(i)); % Random sub-dictionaries
            D_terms{i,p} = D_terms{i,p}*diag(1./sqrt(sum(D_terms{i,p}.^2))); % Normalize columns
        end
        D = D + kron(D_terms(I:-1:1,p));
    end
    normVec = sqrt(sum(D.^2));
    D = D*diag(1./normVec); % Normalize columns. mean(sqrt(sum(D.^2))) approx. sqrt(m)

    % Testing that products with dictionaries give same result
%     y = randn(prod(n),1); % random data
%     teste = D.'*y;
%     teste2 = zeros(m);
%     for p = 1:R
%         teste2 = teste2 + tmprod(reshape(y,n),cellfun(@transpose, D_terms(:,p),'UniformOutput',false),1:I);
%     end
%     teste2 = teste2(:)./normVec.';
%     assert(norm(teste-teste2(:),'fro')<eps*prod(m),'Structured product does not give same result as normal product')

    % Call estimation script
    RC
    
    save(strcat('timeResults_mode_prod_n',sprintf('%.0f_' , n), 'm', sprintf('%.0f_' , m), ...
                'R', num2str(R),'_sparse',num2str(sparse),'_it',num2str(mc_it)), ...
         'time_dense','time_denseT','time_sukro','time_sukroT', ...
         'mean_time_mode',...
         'cost_struct','cost_structT','cost_dense','cost_denseT',...
         'RC_theoretical','RC_theoreticalT','RC_empirical','RC_empiricalT'); 
end


