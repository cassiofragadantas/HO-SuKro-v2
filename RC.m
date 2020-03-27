% Evaluates the Relative Complexity on matrix-vector products for:
% 1) Structured HO-SUKro dictionary: tmprod (tensorlab) or modeprod
%    (homemade) by commenting and uncommenting corresponding lines inside     
%    'for r = 1:cpd_rank' loop.

% 2) Dense matrix-vector problem 

if ~exist('cpd_rank','var')
    cpd_rank = size(D_terms,2); %size(Uhat{1},2);
end

% Parameters
mc_it = 1000;                  % Nb. of Monte-Carlo realizations to be averaged
sparse = false;                 % Sparse data
density = 2/100; %3/prod(m);    % Density of sparse data vector (if sparse=true)
profile_mode_prod = false;      % Profile one instance of mode product

if ~sparse, density = 1; end

%% Theoretical RC
% product D*x
if length(n) == 3
% Brute force: testing all ordering combinations and taking the minimum
    cost_struct = min([n(1)*m(1)*m(2)*m(3) + n(2)*m(2)*n(1)*m(3) + n(3)*m(3)*n(1)*n(2), ...
                       n(1)*m(1)*m(2)*m(3) + n(3)*m(3)*n(1)*m(2) + n(2)*m(2)*n(1)*n(3), ...
                       n(2)*m(2)*m(1)*m(3) + n(1)*m(1)*n(2)*m(3) + n(3)*m(3)*n(1)*n(2), ...
                       n(2)*m(2)*m(1)*m(3) + n(3)*m(3)*n(2)*m(1) + n(2)*m(2)*n(2)*n(3), ...
                       n(3)*m(3)*m(1)*m(2) + n(1)*m(1)*n(3)*m(2) + n(2)*m(2)*n(3)*n(1), ...
                       n(3)*m(3)*m(1)*m(2) + n(2)*m(2)*n(3)*m(1) + n(1)*m(1)*n(3)*n(2)]);

    cost_structT= min([m(1)*n(1)*n(2)*n(3) + m(2)*n(2)*m(1)*n(3) + m(3)*n(3)*m(1)*m(2), ...
                       m(1)*n(1)*n(2)*n(3) + m(3)*n(3)*m(1)*n(2) + m(2)*n(2)*m(1)*m(3), ...
                       m(2)*n(2)*n(1)*n(3) + m(1)*n(1)*m(2)*n(3) + m(3)*n(3)*m(1)*m(2), ...
                       m(2)*n(2)*n(1)*n(3) + m(3)*n(3)*m(2)*n(1) + m(2)*n(2)*m(2)*m(3), ...
                       m(3)*n(3)*n(1)*n(2) + m(1)*n(1)*m(3)*n(2) + m(2)*n(2)*m(3)*m(1), ...
                       m(3)*n(3)*n(1)*n(2) + m(2)*n(2)*m(3)*n(1) + m(1)*n(1)*m(3)*m(2)]);

else
    % The chosen ordering corresponds to sorting the modes wrt the ratio 
    % m(i)/n(i) in decreasing order.
    % This is the heuristic used in tmprod for choosing mode ordering.
    % This is just an heuristic, since it neglects the fact that it is also
    % important to start by a small n (for direct operator, and big n for transpose).
    % Example of configuration in which it
    % doens't work:
    % n = [5 4 2]; m = [10 7.5 3.5]; 
    % Heuristic gives [1,2,3], while best ordering is actually [3,2,1].

    [ratio_sorted, idx] = sort(m./n,2,'descend');

    cost_struct = 0;
    cost_structT = 0;
    for k = 1:length(n)
        if sparse && (k==1) % First mode product is faster for sparse data
            cost_struct = cost_struct + n(idx(k))*density*prod(m(idx(k:end)));
            cost_structT = cost_structT + m(idx(k))*density*prod(n(idx(k:end)));
        else
            cost_struct = cost_struct + n(idx(k))*prod(m(idx(k:end)))*prod(n(idx(1:k-1)));
            cost_structT = cost_structT + m(idx(k))*prod(n(idx(k:end)))*prod(m(idx(1:k-1)));
        end
    end
end
cost_struct = cpd_rank*cost_struct;
cost_structT = cpd_rank*cost_structT;


cost_dense = density*prod(n)*prod(m);
cost_denseT = cost_dense;

RC_theoretical = cost_struct/cost_dense;
RC_theoreticalT = cost_structT/cost_denseT;

%% Practical RC

mean_time_dense = 0;
time_dense = zeros(1,mc_it);
mean_time_denseT = 0;
time_denseT = zeros(1,mc_it);

mean_time_sukro = 0;
time_sukro = zeros(1,mc_it);
mean_time_mode = zeros(size(n));
mean_time_sukroT = 0;
time_sukroT = zeros(1,mc_it);

% product D*x
for k_mc = 1:mc_it
    
    if sparse
        x = sprandn(prod(m),1,density); % sparse random vector
        xt = sprandn(prod(n),1,density); % sparse random vector
    else
        x = randn(prod(m),1); % random vector
        xt = randn(prod(n),1); % random vector
    end
    
    % Dense p
    tic
    y = D*x;
    prod_time = toc;
    mean_time_dense = mean_time_dense + prod_time/mc_it;
    time_dense(k_mc) = prod_time;
    
    % Dense p - Transpose
    tic
    y = D.'*xt;
    prod_time = toc;
    mean_time_denseT = mean_time_denseT + prod_time/mc_it;
    time_denseT(k_mc) = prod_time;

    % ---- SUKRO TIME ----

%     tic
    X = reshape(full(x),m); % tensorize the vector
%     X = tensor(reshape(x,m));
    Y = zeros(n);
    tic
    for r = 1:cpd_rank
        Y = Y + tmprod(X,D_terms(:,r),1:length(m)); % multiplication
%         Y = Y + tmprod_sparse(X,D_terms(:,r),1:length(m)); % multiplication
%         Y = Y + modeprod(X,D_terms(:,r),1:length(m),m); % multiplication
%         Y = Y + modeprod3(X,D_terms{1,r},D_terms{2,r},D_terms{3,r},m); % multiplication
%         Y = Y + ttm(X,D_terms(:,r),1:length(m)); % multiplication
    end
    y2 = Y(:); %vectorize the result
    prod_time = toc;
    mean_time_sukro = mean_time_sukro + prod_time/mc_it;
    time_sukro(k_mc) = prod_time;
    
    % ---- MODE TIME (without permute!) ----
    for mode = 1:length(n)
    X_unfolded = unfold(X,mode);
    tic
        Y = D_terms{mode,1}*X_unfolded; % multiplication
%         Y = D_terms{mode,1}*unfold(X,mode); % multiplication
    prod_time = toc;
    mean_time_mode(mode) = mean_time_mode(mode) + prod_time/mc_it;
    end
    clear X_unfolded
    
    % ---- SUKRO TIME - transpose ----

%     tic
    Xt = reshape(full(xt),n); % tensorize the vector
%     XT = tensor(reshape(xT,n));
    Y = zeros(m);
    tic
    for r = 1:cpd_rank
        Y = Y + tmprod(Xt,D_terms(:,r),1:length(n),'T'); % multiplication
%         Y = Y + tmprod_sparse(X,cellfun(@transpose, D_terms(:,r),'UniformOutput',false),1:length(n)); % multiplication
%         Y = Y + modeprod(X,cellfun(@transpose, D_terms(:,r),'UniformOutput',false),1:length(n),n); % multiplication
%         Y = Y + modeprod3(X,D_terms{1,r}.',D_terms{2,r}.',D_terms{3,r}.',n); % multiplication
%         Y = Y + ttm(X,cellfun(@transpose, D_terms(:,r),'UniformOutput',false),1:length(n)); % multiplication
    end
    y2 = Y(:); %vectorize the result
    prod_time = toc;
    mean_time_sukroT = mean_time_sukroT + prod_time/mc_it;
    time_sukroT(k_mc) = prod_time;
end
RC_empirical = mean_time_sukro/mean_time_dense
RC_theoretical

RC_empiricalT = mean_time_sukroT/mean_time_denseT
RC_theoreticalT

for mode = 1:length(n)
    RC_empirical_mode(mode) = mean_time_mode(mode)/mean_time_dense;
    RC_theoretical_mode(mode) = (n(mode)*prod(m))/cost_dense;
end
RC_empirical_mode
RC_theoretical_mode

%% Profile
if profile_mode_prod
    Y = zeros(n);
    profile on;
    for r = 1:cpd_rank
        Y = Y + tmprod(X,D_terms(:,r),1:length(m)); % multiplication
%         Y = Y + tmprod_sparse(X,D_terms(:,r),1:length(m)); % multiplication
%         Y = Y + modeprod(X,D_terms(:,r),1:length(m),m); % multiplication
%         Y = Y + modeprod3(X,D_terms{1,r},D_terms{2,r},D_terms{3,r},m); % multiplication
%         Y = Y + ttm(X,D_terms(:,r),1:length(m)); % multiplication
    end
    profile off
    profsave(profile('info'),'myprofile_mode_product_results')
end
