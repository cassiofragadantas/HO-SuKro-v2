% This funciton performs the rearrangement defined in [1] with a low memory
% requirement.
% (n1,m1) Dimensions of sub-matrix A
% (n2,m2) Dimensions of sub-matrix B
% So that the resulting matrix is D = kron(A,B) with size (n1*n2, m1*m2)
% and the rearranged version R_D has size (n1*m1,n2*m2)
%
% This inverse rearrangement (as well as it transposed version) has the 
% effect of transforming low rank matrices into sums of Kronecker products.
%
%  [1] C.F. Dantas, M. N. da Costa, R.R. Lopes, "Learning Dictionaries as a
%      sum of Kronecker products", To appear.

function [D] =  reord_inv(R_D,n1,n2,m1,m2,idx)

if nargin > 5 %Rearrangement indexes were provided
    D = reshape(R_D(idx),n1*n2,m1*m2);
else
    D = zeros(n1*n2,m1*m2);
    for j1 = 1:m1
        for i1 = 1:n1
            for j2 = 1:m2
                for i2 = 1:n2
                    % Rearrangement as in equation [3]
                    % The rearranged R_D has size (n1*m1,n2*m2)
                    in_row = i1+(j1-1)*n1;
                    in_col = i2+(j2-1)*n2;
                    out_row  = i2+(i1-1)*n2;
                    out_col  = j2+(j1-1)*m2;
                    D(out_row,out_col) = R_D(in_row,in_col);

                    % Transpose rearrangement
                    % The rearranged R_D has size (n2*m2,n1*m1)
%                     in_col = i1+(j1-1)*n1;
%                     in_row = i2+(j2-1)*n2;
%                     out_row  = i2+(i1-1)*n2;
%                     out_col  = j2+(j1-1)*m2;
%                     D(out_row,out_col) = R_D(in_row,in_col);
                end
            end
        end
    end
end

end