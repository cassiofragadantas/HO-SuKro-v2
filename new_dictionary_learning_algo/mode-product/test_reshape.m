% Goal: switch directly from the unfold of a given mode to the unfold of
% another mode, without using the permute function (more costly than reshape)
%
% Conclusion: cannot go faster than permute. It allows to use sparse
% tensors, but this is not enough to compensate.

% Small
dims = [2 3 4];
T = reshape(1:prod(dims),dims);

% Small 4-D
dims = [2 3 4 2];
T = reshape(1:prod(dims),dims);

% Big
dims = [30 40 50 50];
% T = reshape(1:prod(dims),dims);
T = randn(dims);

% Image size
dims = [12 12 6 4000];
T = reshape(1:prod(dims),dims);
% T = randn(dims);

% Sparse
T = zeros(dims);
T(1,1,1,1) = 1;

n_dims = length(dims);

T1 = unfold(T,1); %T1234
T2 = unfold(T,2);
T3 = unfold(T,3); % T3124

T1 = sparse(T1);

T1423 = reshape(permute(T,[1 4 2 3]),dims(1),[]);
T3412 = reshape(permute(T,[3 4 1 2]),dims(3),[]);
    

%% Generate index vectors
count = 1:prod(dims);
idx_12 = reshape(unfold(reshape(count,dims),2),[],1);
idx_13 = reshape(unfold(reshape(count,dims),3),[],1);

tens = reshape(count,[dims(2) dims(1) dims(3:end)]);
% T1_count = reshape(permute(tens,[2 1 3:n_dims]),dims(1),[]); % equivalent to: unfold(tens,2)
% idx_21 = T1_count(:);
idx_21 = reshape(permute(tens,[2 1 3:n_dims]),[],1);
idx_23 = reshape(permute(tens,[3 2 1 4:n_dims]),[],1);

tens = reshape(count,[dims(3) dims(1:2) dims(4:end)]);
idx_31 = reshape(permute(tens,[2 3 1 4:n_dims]),[],1);
idx_32 = reshape(permute(tens,[3 2 1 4:n_dims]),[],1);


% Checking results
% From T1
tic, T2_test = reshape(T1(idx_12),dims(2),[]);toc 
norm(T2-T2_test,'fro')
T3_test = reshape(T1(idx_13),dims(3),[]); norm(T3-T3_test,'fro')
% From T2
T1_test = reshape(T2(idx_21),dims(1),[]); norm(T1-T1_test,'fro')
T3_test = reshape(T2(idx_23),dims(3),[]); norm(T3-T3_test,'fro')
% From T3
T1_test = reshape(T3(idx_31),dims(1),[]); norm(T1-T1_test,'fro')
T2_test = reshape(T3(idx_32),dims(2),[]); norm(T2-T2_test,'fro')


%% T1 -> T2
tic
T2_test = zeros(dims(2),dims(1)*dims(3));
for k = 1:dims(3)
    T2_test(:,(k-1)*dims(1) + (1:dims(1))) = T1(:,(k-1)*dims(2) + (1:dims(2))).';
end
toc
% norm(T2 - T2_test,'fro') % testing correctness

% If input tensor is more than 3D
tic
T2_test = zeros(dims(2),prod(dims([1 3:end])));
for k = 1:prod(dims(3:end))
    T2_test(:,(k-1)*dims(1) + (1:dims(1))) = T1(:,(k-1)*dims(2) + (1:dims(2))).';
end
toc
% norm(T2 - T2_test,'fro') % testing correctness

%% T1 -> T3
tic
T3_test = reshape(T1,dims(1)*dims(2),dims(3)).';
toc
% norm(T3 - T3_test,'fro') % testing correctness

% If input tensor is more than 3D
tic
T3_test = zeros(dims(3),prod(dims([1:2 4:end])));
for k = 1:prod(dims(4:end))
    T3_test(:,(k-1)*dims(1)*dims(2) + (1:dims(1)*dims(2))) = reshape(T1(:,(k-1)*dims(2)*dims(3) + (1:dims(2)*dims(3))),dims(1)*dims(2),[]).'; % OR reshape(...,[],dims(3))
end
toc
% norm(T3 - T3_test,'fro') % testing correctness


%% T2 -> T3
tic
T3_test = zeros(dims(3),dims(1)*dims(2));
for k = 1:dims(3)
    T3_test(k,:) = reshape(T2(:,(k-1)*dims(1) +(1:dims(1))).',1,[]);
end
toc

% If input tensor is more than 3D
tic
T3_test = zeros(dims(3),prod(dims([1:2 4:end])));
for k = 1:dims(3)
    for kk = 1:prod(dims(4:end))
        T3_test(k,(kk-1)*prod(dims(1:2)) + (1:prod(dims(1:2)))) = reshape(T2(:,(kk-1)*dims(1)*dims(3) + (k-1)*dims(1) +(1:dims(1))).',1,[]);
    end
end
toc

%% T3 -> T1
tic
T1_test = reshape(T3.',dims(1),[]);
toc

% If input tensor is more than 3D
tic
T1_test = zeros(dims(1),prod(dims(2:end)));
for k = 1:prod(dims(4:end))
    T1_test(:,(k-1)*dims(2)*dims(3) + (1:dims(2)*dims(3))) = reshape(T3(:,(k-1)*dims(1)*dims(2)+(1:dims(1)*dims(2))).',dims(1),[]);
end
toc
% norm(T1 - T1_test,'fro') % testing correctness


%% T3 -> T2
tic
T2_test = zeros(dims(2),dims(1)*dims(3));
for k = 1:dims(3)
    T2_test(:,(k-1)*dims(1) + (1:dims(1))) = reshape(T3(k,:),dims(1),dims(2)).';
end
toc

% TODO If input tensor is more than 3D
tic
T2_test = zeros(dims(2),prod(dims([1 3:end])));
for k = 1:dims(3)
    for kk = 1:prod(dims(4:end))
        T2_test(:,(kk-1)*dims(1)*dims(3) + (k-1)*dims(1) +(1:dims(1))) = reshape(T3(k,(kk-1)*prod(dims(1:2)) + (1:prod(dims(1:2)))).',dims(1),dims(2)).';
    end
end
toc
% norm(T2 - T2_test,'fro') % testing correctness


%% Permute
tic
T_test = permute(T,[2:n_dims 1]);
toc

tic
T_test = unfold(T,2);
toc

%% Testing new tmprod
% Small
% dims = [2 3 4 2];
% n = [3 4 2];
% T = reshape(1:prod(dims),dims);
% T1 = unfold(T,1);

% Image size
dims = [12 12 6 40000];
n = [6 6 3];
density = 0.1/dims(end);
T1 = sprand(dims(1),prod(dims(2:end)),density); % random sparse
T = reshape(full(T1),dims);

% Generate index vectors
count = 1:(prod(dims(2:end))*n(1));
idx_12 = reshape(unfold(reshape(count,[n(1) dims(2:end)]),2),[],1);
% idx_13 = reshape(unfold(reshape(count,dims),3),[],1);

count = 1:(prod(dims(3:end))*prod(n(1:2)));
tens = reshape(count,[n(2) n(1) dims(3:end)]);
% T1_count = reshape(permute(tens,[2 1 3:n_dims]),dims(1),[]); % equivalent to: unfold(tens,2)
% idx_21 = T1_count(:);
% idx_21 = reshape(permute(tens,[2 1 3:n_dims]),[],1);
idx_23 = reshape(permute(tens,[3 2 1 4:n_dims]),[],1);

count = 1:(prod(dims(4:end))*prod(n(1:3)));
tens = reshape(count,[n(3) n(1:2) dims(4:end)]);
idx_31 = reshape(permute(tens,[2 3 1 4:n_dims]),[],1);
% idx_32 = reshape(permute(tens,[3 2 1 4:n_dims]),[],1);



U1 = randn(n(1),dims(1));
U2 = randn(n(2),dims(2));
U3 = randn(n(3),dims(3));

tic, teste = tmprod(T,{U1 U2 U3},1:3); toc
tic, teste2 = modeprod3(T,U1, U2, U3,dims); toc
tic, teste3 = modeprod3_sparse(T1,U1, U2, U3,dims); toc
tic, teste3 = modeprod3_sparse(full(T1),U1, U2, U3,dims); toc
tic, teste3 = modeprod3_sparse(T1,U1, U2, U3,dims,idx_12,idx_23,idx_31); toc
norm(teste3 - unfold(teste,1),'fro')

% Mode product gives same result for different permutations of remaining
% modes (as long as first mode is the correct one).
teste = tmprod(T,U1,1);
teste2 = tmprod(permute(T,[1 3 2 4]),U1,1); teste2 = permute(teste2,[1 3 2 4]);

% Profile
profile on
% T1 = modeprod3_sparse(full(T1),U1, U2, U3,dims);
teste = modeprod3_sparse(T1,U1, U2, U3,dims,idx_12,idx_23,idx_31);
profile off
profsave(profile('info'),'myprofile_results')

