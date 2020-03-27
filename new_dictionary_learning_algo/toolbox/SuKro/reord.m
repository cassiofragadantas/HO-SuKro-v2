% This funciton performs the rearrangement defined in [1] with a low memory
% requirement.
% (n1,m1) Dimensions of sub-matrix A
% (n2,m2) Dimensions of sub-matrix B
% So that the resulting matrix is D = kron(A,B) with size (n1*n2, m1*m2)
% and the rearranged version R_D has size (n1*m1,n2*m2)
%
% This rearrangement (as well as it transposed version) has the effect of
% transforming sums of Kronecker products into low rank matrices.
%
%  [1] C.F. Dantas, M. N. da Costa, R.R. Lopes, "Learning Dictionaries as a
%      sum of Kronecker products", To appear.

function [R_D] =  reord(D,n1,n2,m1,m2,idx)

if nargin > 5 %Rearrangement indexes were provided
    R_D = reshape(D(idx),m1*n1,m2*n2); % Rearrangement as in equation [3]
    %R_D = reshape(D(idx),m2*n2,m1*n1); % Transpose rearrangement
else
    R_D = zeros(n1*m1,n2*m2); % Rearrangement as in equation [3]
    %R_D = zeros(n2*m2,n1*m1); % Transpose rearrangement
    for j1 = 1:m1
        for i1 = 1:n1
            for j2 = 1:m2
                for i2 = 1:n2
                    % Rearrangement as in equation [3]
                    % The rearranged R_D has size (n1*m1,n2*m2)
                    out_row = i1+(j1-1)*n1;
                    out_col = i2+(j2-1)*n2;
                    in_row  = i2+(i1-1)*n2;
                    in_col  = j2+(j1-1)*m2;
                    R_D(out_row,out_col) = D(in_row,in_col);

                    % Transpose rearrangement
                    % The rearranged R_D has size (n2*m2,n1*m1)
%                     out_col = i1+(j1-1)*n1;
%                     out_row = i2+(j2-1)*n2;
%                     in_row  = i2+(i1-1)*n2;
%                     in_col  = j2+(j1-1)*m2;
%                     R_D(out_row,out_col) = D(in_row,in_col);
                end
            end
        end
    end
end

end