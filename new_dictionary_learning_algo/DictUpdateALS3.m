% >>> Alternating Least-Squares for optimizing HO-SuKro terms. <<<
% This version optimizes all rank terms (p=1:R) simultaneously.
%
% Difference w.r.t ALS2 : uses the tensorial mode-product for the product
% between D and Y (instead of D and X as in ALS2). Theoretically, this
% should be faster in most configurations, but this did not translate into
% practical speedup.
%
% Parameters :
% - Sizes of factors D{i,p} is nixmi for any p (stocked in memory)
%       n = [n1 n2 n3]; % size I
%       m = [m1 m2 m3];
% - N : Number of training samples
% - R : Number of Kronecker summing terms
% - I = length(n) : Number of modes

% addpath ../tensorlab_2016-03-28/

%% Optimizing dictionary, given X and Y
iorder = [2 1 3];%1:I;

% Convergence measures
N_iter = 10; % maximum number of iterations
obj = zeros(1,N_iter);

converged = false;
tol = 1e-1*sqrt(m);
if k == iternum, tol = 1e-3*sqrt(m); end % Better accuracy on the last iteration
diff = zeros(I,R); % Frobenius norm of update on each D_ip
k_ALS = 0;

% D_ip_old = cell(length(n),params.alpha);


D_ipT = cellfun(@transpose, D_ip,'UniformOutput',false);

%Calculate all DTD exept for first i, since it will not be used on the
%first iteration and will be directly updated.
% DTD2 = cell(I,R,R);
% for i = iorder(2:end)
%     DTD2{i} =  cellfun(@(x,y) x.'*y, D_ip([1:i0-1 i0+1:I],p2),D_ip([1:i0-1 i0+1:I],p1), 'UniformOutput',false);
% end

% for k_ALS = 1:N_iter
while ~converged, k_ALS = k_ALS + 1;
fprintf('%4d,',k_ALS);
for i0 = iorder
    
    By = cell(1,R);
    for p = 1:R % All indexes
%         By{p} = unfold(tmprod(Y,D_ipT([1:i0-1 i0+1:I],p),[1:i0-1 i0+1:I]),i0); 
        By{p} = unfold(tmprod(Y,cellfun(@transpose, D_ip([1:i0-1 i0+1:I],p),'UniformOutput',false),[1:i0-1 i0+1:I]),i0); 
        By{p} = -By{p}*unfold(X,i0).';
    end
    
    UUT = cell(R,R);
    for p1 = 1:R % All indexes
        for p2 = 1:R % All indexes
            DTD = cellfun(@(x,y) x.'*y, D_ip([1:i0-1 i0+1:I],p2),D_ip([1:i0-1 i0+1:I],p1), 'UniformOutput',false); % D_ip.'*D_ip for all i and p
            UUT{p1,p2} = unfold(tmprod(X,DTD,[1:i0-1 i0+1:I]),i0)*unfold(X,i0).';
        end
    end
    A = cell2mat(UUT);
    By = cell2mat(By);
    
    Di0 =  -(A.'\By.').';
    
    for p0 = 1:R
        diff(i0,p0) = norm(D_ip{i0,p0}-Di0(:,m(i0)*(p0-1) + (1:m(i0))),'fro');
        D_ip{i0,p0} = Di0(:,m(i0)*(p0-1) + (1:m(i0)));
%         D_ipT{i0,p0} = D_ip{i0,p0}.';
    end

end

% stop Criterion
diff
if all(mean(diff,2) < tol.')
    converged = true;
    % disp(['Total nº of iterations: ' num2str(iter)]);
end

% Other stop criterion (potentially more costly)
% Calculate the objective function
% Y_r = zeros([n N]);
% for p=1:R
%     %Y_r = Y_r + tmprod(X,D_ip(1:I,p),fliplr(1:I)); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
%     Y_r = Y_r + tmprod(X,D_ip(1:I,p),1:I); % Y = D*X, gives sames results as Y = Y + kron(D_ip_oracle(1:I,p))*X(:);
% end
% 
% obj(k_ALS) = norm(Y(:)-Y_r(:),'fro');

% Difference in complete dictionary
% D_structured = zeros(size(D));
% for p = 1:R %TODO get rid of kron. Use directly the D_ip
% %         for i=1:I
% %             D_ip{i,p} = normc(D_ip{i,p});
% %         end
%     D_structured = D_structured + kron(D_ip(I:-1:1,p));
% end
% D_structured_old = zeros(size(D));
% for p = 1:R %TODO get rid of kron. Use directly the D_ip
% %         for i=1:I
% %             D_ip{i,p} = normc(D_ip{i,p});
% %         end
%     D_structured_old = D_structured_old + kron(D_ip_old(I:-1:1,p));
% end
% norm(D_structured-D_structured_old,'fro')

end
