% Function that calculates the column norms of an HO-SuKro dictionary,
% without unfolding kronecker products (i.e. manipulating only the blocks
% D_ip). Actually,  the columns are unfolded individually in a for loop.
% This function is to be used if memory low memory usage is required (such
% that storing the full dictionary in memory is not possible).
%
%   Author: Cassio Fraga Dantas
%   Date: 09 March 2019

function normVec = normColsKron(D_ip,n,m)
I = size(D_ip,1);
R = size(D_ip,2);

normVec = normColsKron_recursive_faster(D_ip,n,m,[],I,R,num2cell(ones(1,R)));
% normVec = normColsKron_recursive(D_ip,n,m,[],I,R); % Gives same result

end


% Recursive function to replace the nested for-loops on m(i) with
% i={1,...,I}. The advantage is that it works for arbitrary I (with nested 
% loops, one need to explicitly hardcode a certain number of nested loops).
function normVec = normColsKron_recursive(D_ip,n,m,k_prev,I,R)

normVec = [];

for k = 1:m(1)
    
    if length(m) > 1 % recursive call
        normVec = [normVec normColsKron_recursive(D_ip,n,m(2:end),[k_prev k],I,R)];
    else % base case
        
        % Unfold the corresponding column (from scratch)
        col = zeros(prod(n),1);
        k_all = [k_prev k];
        for p = 1:R
            col_p = 1;
            for i = 1:I
                col_p = kron(col_p,D_ip{i,p}(:,k_all(i)));
            end
            col = col + col_p;
        end
        normVec = [normVec norm(col,2)];

    end
end

end

% Same as normColsKron_recursive, but without recalculating the column
% every time from scratch. Here, we resuse the column unfolding until the 
% previous level (mode).
function normVec = normColsKron_recursive_faster(D_ip,n,m,k_prev,I,R, col_prev)

normVec = [];
col_current = cell(size(col_prev));

for k = 1:m(1)
    % Continue unfolding column until current level (mode)
    for p = 1:R
        col_current{p} = kron(col_prev{p},D_ip{length(k_prev)+1,p}(:,k));
    end
    
    if length(m) > 1 % recursive call
        normVec = [normVec normColsKron_recursive_faster(D_ip,n,m(2:end),[k_prev k],I,R,col_current)];
    else % base case
        % Sum over R and calculate column norm
        normVec = [normVec norm(sum(cell2mat(col_current),2),2)];

    end
end

end
