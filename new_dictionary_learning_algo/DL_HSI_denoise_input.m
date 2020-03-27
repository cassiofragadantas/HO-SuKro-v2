function DL_HSI_denoise_input(imnum, exp_type, algo_type, iternum, sigma, mc_it,samples_training,blocksize,blocksize_m,gain)
% Same as DL_image_denoise_3D_input but using hyperspectral images.
% This function is to be used coupled with run_all.m script.
%
% This DEMO is based on the image denoising demo available at the KSVDBOX
% package. 
%
%  Reference:
%  [1] M. Elad and M. Aharon, "Image Denoising via Sparse and Redundant
%      representations over Learned Dictionaries", the IEEE Trans. on Image
%      Processing, Vol. 15, no. 12, pp. 3736-3745, December 2006.
%
% The dictionary learning algorithm is exchanged by the SuKro (Sum of Kroneckers)
% dictionary learning technique (sum_separable_dict_learn.m)
%
%  [2] C.F. Dantas, M.N. da Costa and R.R. Lopes, "Learning Dictionaries as
%       a sum of Kronecker products"

%  DEMO_image_denoise reads an image, adds random white noise and denoises it
%  The input and output PSNR are compared, and the
%  trained dictionary is displayed.
%
%  To run the demo, type DEMO_image_denoise from the Matlab prompt.
%
%  See also image_denoise for the denoising algorithm.


%  ORIGINAL DEMO BY:
%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  August 2009
%
%  ADAPTED BY:
%  Cassio Fraga Dantas
%  DSPCom Laboratory - Unicamp
%  Campinas, Brasil
%  cassio@decom.fee.unicamp.br
%
%  September 2016

% rng(100); disp('/!\ /!\ /!\ RANDOM SEED: ON /!\ /!\ /!\')

%% For Igrida simulations (code is compiled, i.e. isdeployed == true)
if isdeployed
    % Convert input strings into numbers
    imnum = eval(imnum);
    exp_type = eval(exp_type);
    algo_type = eval(algo_type);
    iternum = eval(iternum);
    sigma = eval(sigma);
    mc_it = eval(mc_it);
    samples_training = eval(samples_training);
    blocksize = eval(blocksize);
    blocksize_m = eval(blocksize_m);
    gain = eval(gain);
end

%% Downloading OMP and KSVD Toolboxes if necessary
if ~isdeployed
    FS=filesep;
    pathstr = fileparts(which('DL_HSI_denoise_input'));
    addpath([pathstr,filesep,'..']); %addpath([pathstr,filesep,'..',filesep,'misc']);
    addpath([pathstr,filesep,'..',filesep,'tensorlab_2016-03-28']);
    addpath([pathstr,filesep,'HSIdata']);
    addpath([pathstr,filesep,'sparse-coding']);
    addpath([pathstr,filesep,'mode-product']);

    SuKro_path = [pwd,FS,'toolbox/SuKro'];
    addpath(SuKro_path);

    % KSVD and OMP toolboxes are assumed to be already installed
    % KSVD_path = [pwd,FS,'toolbox'];
    addpath([pathstr,filesep,'toolbox',filesep,'ksvdbox']);
    addpath([pathstr,filesep,'toolbox',filesep,'ompbox']);
end

%% Prompt user %%
disp(' ');
disp('***********************  SuKro Dictionary Denoising Demo  ***********************');
disp('*                                                                               *');
disp('* This demo reads an image, adds random Gaussian noise, and denoises the image  *');
disp('* using a dictionary which is the sum of a few separable terms. The function    *');
disp('* displays the original, noisy, and denoised images, and shows the resulting    *');
disp('* trained dictionary.                                                           *');
disp('*                                                                               *');
disp('*********************************************************************************');

%dirname = fullfile(pathstr, 'images', '*.tiff');
%imglist = dir(dirname);
imglist(1).name = 'SanDiego';
imglist(2).name = 'SanDiego-truncated';
imglist(3).name = 'SanDiego-reshaped-2D';
imglist(4).name = 'SanDiego-small';
imglist(5).name = 'SanDiego-reshaped-2D-small';
imglist(6).name = 'Washington';
imglist(7).name = 'Washington-truncated';
imglist(8).name = 'Washington-reshaped-2D';
imglist(9).name = 'Washington-small';
imglist(10).name = 'Urban';
imglist(11).name = 'Urban-small';
imglist(12).name = 'Houston';
imglist(13).name = 'Houston-truncated-left';
imglist(14).name = 'Houston-truncated-center';
imglist(15).name = 'Houston-small';

disp('  Available test images:');
disp(' ');
for k = 1:length(imglist)   
  fprintf('  %d. %s\n', k, imglist(k).name);
end
fprintf('  %d. All images', length(imglist)+1);
disp(' ');

% imnum = 0;
% while (~isnumeric(imnum) || (rem(imnum,1)~=0) || imnum<1 || imnum>length(imglist)+1)
%   imnum = input(sprintf('  Image to denoise (%d-%d): ', 1, length(imglist)), 's');
%   imnum = sscanf(imnum, '%d');
% end
%
% imnum is given as input parameter
fprintf('  Image to denoise: %d.\n', imnum);

total_images = 1;
if imnum == length(imglist)+1, total_images = length(imglist), end

fprintf('\n\n  Choose the experiment type:\n');
fprintf('\n  1. Single-run (a few minutes)\n  2. Complete   (a few hours)\n');
% exp_type = 0;
% while (~isnumeric(exp_type) || (rem(exp_type,1)~=0) || exp_type<1 || exp_type>2)
%   exp_type = input(sprintf('  Experiment type (1 or 2): '), 's');
%   exp_type = sscanf(exp_type, '%d');
% end
% exp_type is given as input parameter
fprintf('  Experiment type: %d.\n', exp_type);

fprintf('\n\n  Choose the algorithm:\n');
fprintf('\n  1. SuKro\n  2. HO-SuKro\n  3. K-SVD\n  4. ODCT\n');
% algo_type = 0;
% while (~isnumeric(algo_type) || (rem(algo_type,1)~=0) || algo_type<1 || algo_type>4)
%   algo_type = input(sprintf('  Algorithm type (1, 2, 3 or 4): '), 's');
%   algo_type = sscanf(algo_type, '%d');
% end
% algo_type is given as input parameter
fprintf('  Algorithm type: %d.\n', algo_type);
params.algo_type = algo_type;

for image = 1:total_images 

if image > 1, imnum = image; end
if imnum == length(imglist)+1, imnum = 1; end
% imgname = fullfile(pathstr, 'images',imglist(imnum).name);
imgname = imglist(imnum).name;

%% Simulation parameters %%
% sigma is given as input argument
fprintf('\nSIGMA = %d\n',sigma)
% mc_it = 5; % Number of Monte-Carlo iterations

% samples_training = 40000; %[1000 2000 5000 10000 20000]; % Number of training samples
params.iternum = iternum; %20;
params.blocksize = blocksize; %[6 6 3];
% M3 = 6;
% params.dictsize = 4*params.blocksize(1)*params.blocksize(2)*M3;
M3=1; if length(blocksize_m)>=3, M3 = blocksize_m(3);end
params.dictsize = blocksize_m(1)*blocksize_m(2)*M3;

show_dict = false;
save_dict = false;

params.sigma = sigma;
params.maxval = 256*128 - 1; %255;
params.memusage = 'high';
params.my_omp_training = false; % MODIF: To use homemade OMP functions in denoising
if params.my_omp_training, fprintf('\n\n !!! \n USING HOMEMADE OMP DURING TRAINING\n !!!\n\n'); end
params.my_omp_denoise = false; % MODIF: To use homemade OMP functions in denoising
if params.my_omp_denoise, fprintf('\n\n !!! \n USING HOMEMADE OMP FOR DENOISING\n !!!\n\n'); end
params.gain = gain;

% Adding Low-rank
params.lowrank = 1; % 0 (or false): Usual simulation, only Dictionary Learning is used
                    % Otherwise, this variable should contain the number of
                    % iterations of Low-rank + DL steps (true gives 1 iteration)
params.lowrank_left = true; % true:  only the left SVD term is denoised, 
                            %        column-by-column, each leading to a 
                            %        2D image denoised separately (used in Whispers' paper)
                            % false: a 3D HSI is rebuilt from the low-rank 
                            %        approximation and denoised using 3D patches
if params.lowrank, fprintf('\n\n !!! \n USING LOW-RANK ASSUMPTION\n !!!\n\n'); end

if (exp_type == 1)  % Single-run experiment
    if algo_type == 1
        % SuKro
%         alpha = 250;
        params.u = 1; alpha = 0.02; % 0.05: rank1. 0.02: rank3. 0.01: rank6
    else
        % HO-SuKro
        alpha = 3;
    end
else                % Complete experiment
    if algo_type == 1
        alpha = logspace(log10(100),log10(600),50); % Nuclear norm regularization coefficient
    elseif algo_type == 2
%         alpha = [1 2 3 4 5]; % Number of separable terms
        alpha = 1:2:5; % Number of separable terms
        fprintf('\n !!!! alpha = [1 3 5] only !!!!\n');
        %alpha = [1 3]; % Number of separable terms
    else % KSVD or ODCT. This parameter is useless
        alpha = 1;
    end
end

%im = imread(imgname);
im = loadHSI(imgname);
im = double(im);

% Subdictionaries (Kronecker factors) dimensions
params.kro_dims.N1 = params.blocksize(1);
params.kro_dims.N2 = params.blocksize(2);
% params.kro_dims.N3 = params.blocksize(3);
params.kro_dims.N3 = 1; if length(blocksize_m)>=3, params.kro_dims.N3 = params.blocksize(3);end
params.kro_dims.M1 = blocksize_m(1); %sqrt(params.dictsize/M3);
params.kro_dims.M2 = blocksize_m(2); %sqrt(params.dictsize/M3);
params.kro_dims.M3 = M3;


if (isfield(params,'lowrank_left') && params.lowrank_left) %2D-data
    params.kro_dims.N = [params.kro_dims.N1 params.kro_dims.N2];
    params.kro_dims.M = [params.kro_dims.M1 params.kro_dims.M2];
else %3D-data
    params.kro_dims.N = [params.kro_dims.N1 params.kro_dims.N2 params.kro_dims.N3];
    params.kro_dims.M = [params.kro_dims.M1 params.kro_dims.M2 params.kro_dims.M3];
end

% Teste, denoising each color separately
%im = im(:,:,3); % TODO teste
%params.blocksize = [6 6];
%params.dictsize = 4*params.blocksize(1)*params.blocksize(2);

% results to store
PSNR = zeros(size(alpha));
PSNR_per_band = zeros(length(alpha),size(im,3));
SNR_in = zeros(size(alpha)); % SNR input
SNR = zeros(size(alpha));
SNR_per_band = zeros(length(alpha),size(im,3));

if (isfield(params,'lowrank') && params.lowrank)
    if ~params.lowrank_left % denoising whole image
        SNR_perIt_lr = zeros(length(alpha),params.lowrank+1);
        SNR_perIt_sparse = zeros(length(alpha),params.lowrank+1);
        approx_rank = zeros(length(alpha),params.lowrank+1);
    else % denoising left factor only
        SNR_perIt_lr = zeros(length(alpha),1);
        SNR_perIt_sparse = zeros(length(alpha),1);
        approx_rank = zeros(length(alpha),1);
    end
end

exec_times = cell(size(alpha));

if params.algo_type == 1 % For SuKro results (when using nuclear norm)
    params.rank_Dmod = cell(size(alpha));
end

%% Initial dictionary
% Random data - doesn't help
% params.initdict = 'data';

% 2D-ODCT and equidistant data samples in 3rd mode (spectral)
if false
    I = length(blocksize_m);
    params.initdict = odctndict(params.blocksize(1:2),params.kro_dims.M(1:2), 2);

    % samples
    avg_block_dims = params.blocksize; % doens't have to be blocksize. Only last mode size has to coincide.
    ids = cell(I,1);
    [ids{:}] = reggrid(size(im)-avg_block_dims+1, params.kro_dims.M(3), 'eqdist');
    blocks = sampgrid(im,avg_block_dims,ids{:});
    blocks = remove_dc(blocks,'columns');

    spectral_atoms = zeros(params.kro_dims.N(3),params.kro_dims.M(3));
    for k_block = 1:params.kro_dims.M(3)
        % Averaging over two first modes
        spectral_atoms(:,k_block) = mean(mean(reshape(blocks(:,k_block),blocksize),1),2);
    end

    params.initdict = kron(normcols(spectral_atoms),params.initdict);

    % Initialize the kronecker terms D_ip
    params.odct_factors = cell(I,1);
    for i_idx=1:I
        params.odct_factors{i_idx,1} =  odctdict(params.kro_dims.N(i_idx),params.kro_dims.M(i_idx));
    end
end

%% Run Simulations
for k_mc = 1:mc_it
% same noise
n = randn(size(im)) * sigma;
imnoise = im + n;

for ktrain = 1:length(samples_training)
for k = 1:length(alpha)
%% Generate noisy image %%
fprintf('\n>>>>>>>>>> Running chosen parameters - %d of %d <<<<<<<<<<\n',k,length(alpha));
disp('Generating noisy image...');

%n = randn(size(im)) * sigma;
%imnoise = im + n;

%% Denoise %%
params.x = imnoise;
params.trainnum = samples_training(ktrain);
params.alpha = alpha(k);

if (isfield(params,'lowrank') && params.lowrank)
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    disp('!!!!!!!!!!!!!!! MODIF !!!!!!!!!!!!!')
    disp('!!!!!!!!!!! LOW-RANK TEST !!!!!!!!!')
    disp('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    params.im = im; % DEBUG Just for debugging
    params.sigma0 = std(imnoise(:)-im(:));
    
    %% Pre-processing (LR)
    SNR_perIt_sparse(k,1) = 20*log10( norm(im(:)) / norm(im(:)-imnoise(:)))

    exec_times{k}.total = tic; % measure total execution time
    exec_times{k}.svd = tic; % measure low-rank approximation time
    max_rank = 30;
%     s_end = sqrt(eigs(reshape(imnoise,size(im,1)*size(im,2),size(im,3)).'*reshape(imnoise,size(im,1)*size(im,2),size(im,3)),1,'sm'));
%     [U, S, V] = svds(reshape(imnoise,size(im,1)*size(im,2),size(im,3)),max_rank); % Around 30x slower than svd 'econ'
    [U, S, V] = svd(reshape(imnoise,size(im,1)*size(im,2),size(im,3)),'econ'); U=U(:,1:max_rank);S=S(:,1:max_rank); V=V(:,1:max_rank);
    exec_times{k}.svd = toc(exec_times{k}.svd);
    
    % ---- Rank selection ----
    
    % Oracle
%     SNR_lr = zeros(1,max_rank);
%     for k_rank = 1:max_rank     % Calculate SNR by rank
%         params.x = U(:,1:k_rank)*S(1:k_rank,1:k_rank)*V(:,1:k_rank).';
%         SNR_lr(k_rank) = 20*log10( norm(im(:)) / norm(im(:)-params.x(:)));
%     end
%     % Choose optimal rank
%     [~, idx] = max(SNR_lr);
%     idx = idx + 5;
    
    sing_vals = diag(S);
    % Empirical Criterion 1 - gives result similar to oracle idx+5
%     idx = find((sing_vals(1:max_rank-1) - sing_vals(2:max_rank))./sing_vals(2:max_rank)<5e-3,1); %5e-3, 1e-2
    % Empirical Criterion 2
    idx = find((sing_vals(1:max_rank-1) - sing_vals(max_rank))./sing_vals(max_rank)<3e-2,1); % 3e-2, 5e-2

    if params.lowrank_left
    % Only the left SVD term (U) is denoised, column-by-column, each being
    % reshaped as a 2 image and denoised via Dictionary Learning using
    % 2D-patches for training. (Used in Whispers 2019 paper).
        
%         tic, [U_in S_in V_in] = svds(reshape(im,size(im,1)*size(im,2),size(im,3)),max_rank); toc % Around 30x slower than svd 'econ'
        tic, [U_in S_in V_in] = svd(reshape(im,size(im,1)*size(im,2),size(im,3)),'econ'); U_in=U_in(:,1:max_rank);S_in=S_in(:,1:max_rank); V_in=V_in(:,1:max_rank);toc
%         noise_sing_vals = svd(reshape(n,size(n,1)*size(n,2),size(n,3)));
        signs = sign(sum(U.*U_in)); % determines if vectors are reversed or not
        
        % TESTS - Noise
%         20*log10(1/norm(U(:,2)-U_in(:,2))) % SNR of this column
%         20*log10(S_in(1,1)/S_in(2,2)) % close to SNR_1 - SNR_2
%         SNR_inCol(1) - 20*log10(S_in(1,1)./diag(S_in)).' % Gives the predicted SNR by column
%         % Using clean U and noisy S and V (this is the limit of the method)
%         U_in(:,1:idx)*diag(signs(1:idx))*S(1:idx,1:idx)*V(:,1:idx).';
        
        params.x = reshape(U(:,1:idx),size(im,1),size(im,2),idx);
        
        % ---- SNR after LR pre-processing ----
        
        % column by column
        SNR_inCol = zeros(1,idx);
        % Oracle
        sigmas = zeros(1,idx);
        for k_col = 1:idx
%             SNR_inCol(k_col) = max(20*log10(1/norm(U(:,k_col)-U_in(:,k_col))), ...
%                              20*log10(1/norm(U(:,k_col)+U_in(:,k_col))) );
            SNR_inCol(k_col) = 20*log10( 1/norm(U(:,k_col)-signs(k_col)*U_in(:,k_col)) );
            sigmas(k_col) = std(U(:,k_col)-signs(k_col)*U_in(:,k_col));
        end
        % Empirical
        sigmas = sing_vals(max_rank)./(sqrt(size(U,1))*sing_vals(1:idx)).'; % Underestimates at final columns
%         sigmas2 = sing_vals(max_rank)./(sqrt(size(U,1))*(sing_vals(1:idx) - sing_vals(max_rank))).'; % Bad! overestimates a lot
%         sigmas = sing_vals(idx)./(sqrt(size(U,1))*sing_vals(1:idx)).'; % Underestimates at final columns

        
        % global
%         SNR_perIt_lr(k,1) = 20*log10( norm(U_in(:,1:idx),'fro')/norm(U(:,1:idx)-U_in(:,1:idx)*diag(signs(1:idx)),'fro') ); % Not considering S
        SNR_perIt_lr(k,1) = 20*log10( norm(U_in(:,1:idx)*S_in(1:idx,1:idx),'fro')/norm(U(:,1:idx)*S(1:idx,1:idx)-U_in(:,1:idx)*S_in(1:idx,1:idx)*diag(signs(1:idx)),'fro') );
        approx_rank(k,1) = idx
                
        %% Sparse step
        U_out = U(:,1:idx);
        SNR_outCol = zeros(1,idx);
        exec_times{k}.training = zeros(1,idx);
        exec_times{k}.denoising = zeros(1,idx);
        if params.algo_type == 1 % For SuKro results (when using nuclear norm)
            params.rank_Dmod{k} = zeros(1,idx);
        end
        
        % ---- Denoising all columns (learn single dictionary) ----
%         params.maxval = 1/(sqrt(size(U,1))); % Chosen after tests calibrating lambda
%         params.sigma = sigmas;
%         [imout, dict, dict_struct, normVec, exec_times_all] = image_denoise_various_noise(params); % Alg. 1
%         exec_times{k}.training = exec_times_all.training;
%         exec_times{k}.denoising = exec_times_all.denoising;
%         SNR_perIt_sparse(k,1) = 20*log10( norm(U_in(:,1:idx)*S_in(1:idx,1:idx),'fro')/norm(U_out(:,1:idx)*S(1:idx,1:idx)-U_in(:,1:idx)*S_in(1:idx,1:idx)*diag(signs(1:idx)),'fro') );
%         U_out = reshape(imout,size(im,1)*size(im,2),idx);

        % ---- Denoising column by column ----
%         start_idx = find(sqrt(size(U,1))*sigmas>1e-2,1); % Index of first column below 40dB SNR
%         U_out(:,1:start_idx-1) = U(:,1:start_idx-1); % High-SNR columns are not denoised
        start_idx = 1;
        for k_col = start_idx:idx
            params.im = signs(k_col)*U_in(:,k_col); % Just for debug and calibration of lambda
            params.x = reshape(U(:,k_col),size(im,1),size(im,2));
            params.maxval = 1/(sqrt(size(U,1))); % Chosen after tests calibrating lambda
            params.sigma = sigmas(k_col);
            
            % ---- Denoising this column ----
            [imout, dict, dict_struct, normVec, exec_times_col] = image_denoise_lr(params); % Alg. 1
            
            
            exec_times{k}.training(k_col) = exec_times_col.training;
            exec_times{k}.denoising(k_col) = exec_times_col.denoising;
            if params.algo_type == 1 % For SuKro results (when using nuclear norm)
                reord_dict = reord(dict_struct,params.kro_dims.N1,params.kro_dims.N2,params.kro_dims.M1,params.kro_dims.M2);
                params.rank_Dmod{k}(k_col) = rank(reord_dict,norm(dict)*2e-7);
            end

            % ---- Output SNR ----
            U_out(:,k_col) = imout(:);
            SNR_inCol
            SNR_outCol(k_col) = 20*log10( 1/norm(imout(:)-signs(k_col)*U_in(:,k_col)) )
            
            % ---- Tests : Initialization ----
            
            % Test initialiser prochaine colonne avec precedente
            params.initdict = dict; 
            params.initdict_struct = dict_struct; params.normVec = normVec;
            % Test initialiser prochaine colonne avec première
%             if k_col==1
%                 params.initdict = dict;
%                 params.initdict_struct = dict_struct; params.normVec = normVec;
%             end
            % Test tout débruiter avec dictionnaire de premiere colonne
%             params.initdict = dict;
%             params.initdict_struct = dict_struct; params.normVec = normVec;
%             params.algo_type = 4; % Needs to be reset when using SuKro
        end
        
        % global
%         SNR_perIt_sparse(k,1) = 20*log10( norm(U_in(:,1:idx),'fro')/norm(U_out(:,1:idx)-U_in(:,1:idx)*diag(signs(1:idx)),'fro') ) % Not considering S
        SNR_perIt_sparse(k,1) = 20*log10( norm(U_in(:,1:idx)*S_in(1:idx,1:idx),'fro')/norm(U_out(:,1:idx)*S(1:idx,1:idx)-U_in(:,1:idx)*S_in(1:idx,1:idx)*diag(signs(1:idx)),'fro') );
        
        imout = U_out*S(1:idx,1:idx)*V(:,1:idx).';
%         imout = U_out*S(1:idx,1:idx)*diag(signs(1:idx))*V_in(:,1:idx).'; % Testing: oracle V
        imout = reshape(imout,size(im));
        
        params.maxval = 256*128 - 1; %255;
        try params = rmfield(params,{'initdict','initdict_struct','normVec'}); catch end
    else
    % The full 3D HSI is rebuilt from the low-rank approximation and 
    % denoised via Dictionary Learning using 3D patches.
    % (Not used in any paper, as it doesn't improve results significantly,
    %  neither w.r.t. only DL denoising, nor w.r.t. simply low-rank approx)
    
        params.x0 = imnoise;

        params.x = reshape(U(:,1:idx)*S(1:idx,1:idx)*V(:,1:idx).',size(im));

        % SNR out - after LR pre-processing
        SNR_perIt_lr(k,1) = 20*log10( norm(im(:)) / norm(im(:)-params.x(:)))
        approx_rank(k,1) = idx

        params.sigma = std(params.x(:)-im(:));

        %% Iterations (DL + LR)
        for k_outer_loop = 1:params.lowrank

    %         [imout, dict, dict_struct, normVec, exec_times{k}] = image_denoise(params); % Alg. 1
            % Alg. 2 or 3 (or even 1)
            [imout, dict, dict_struct, normVec, exec_times{k}] = image_denoise_lr(params); % Uses the original noisy image in the final averaging

            SNR_perIt_sparse(k,k_outer_loop+1) = 20*log10( norm(im(:)) / norm(im(:)-imout(:)))
            
            % TEST - low-rank pos-processing (SVD truncation)
            max_rank = 20;
            [U, S, V] = svds(reshape(imout,size(im,1)*size(im,2),size(im,3)),max_rank);
            % Calculate SNR by rank
            SNR_lr = zeros(1,max_rank);
            for k_rank = 1:max_rank
                imout = U(:,1:k_rank)*S(1:k_rank,1:k_rank)*V(:,1:k_rank).';
                SNR_lr(k_rank) = 20*log10( norm(im(:)) / norm(im(:)-imout(:)));
            end
            SNR_lr
            % Choose optimal rank
            [~, idx] = max(SNR_lr);
    %         idx = idx + 1;
            imout = reshape(U(:,1:idx)*S(1:idx,1:idx)*V(:,1:idx).',size(im));

            % SNR out - after this iteration
            SNR_perIt_lr(k,k_outer_loop+1) = 20*log10( norm(im(:)) / norm(im(:)-imout(:)))
            approx_rank(k,k_outer_loop+1) = idx

            params.sigma = std(imout(:)-im(:))
            params.x = imout; % Commenting this line gives BUG version
        end
        params = rmfield(params,'x0');
    end
    params = rmfield(params,'im');
    exec_times{k}.total = toc(exec_times{k}.total);
else
    [imout, dict, dict_struct, normVec, exec_times{k}] = image_denoise(params);
end


%% Show results (single-run experiment) %%
% if (exp_type == 1)  % Single-run experiment
if (false)  % Single-run experiment
    % Show dictionary - only for 2-D
    if(show_dict) % Caution: This might be slow.
        cd toolbox/ksvdbox/private
        for k_atom=1:params.dictsize
            subplot(ceil(sqrt(params.dictsize)),ceil(sqrt(params.dictsize)),k_atom)
            imshow(imnormalize(reshape(dict(:,k_atom),params.blocksize)))
        end
        cd ../../../
    end
    
    figure; ax = subplot(1,1,1); imshow(im(:,:,1)/params.maxval);
    title(ax,'Original image'); drawnow

    figure; ax = subplot(1,1,1); imshow(imnoise(:,:,1)/params.maxval); 
    title(ax,sprintf('Noisy image, PSNR = %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imnoise(:))) ));

    figure; ax = subplot(1,1,1); imshow(imout(:,:,1)/params.maxval);
    title(ax,sprintf('Denoised image, PSNR: %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:))) ));
end

%% Quality measures
% PSNR
PSNR(k) = 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
% PSNR per spectral band
noise_per_band = im-imout;
noise_per_band = sqrt(sum(sum(noise_per_band.^2),2)); % noise norm per band
PSNR_per_band(k,:) = squeeze(20*log10(params.maxval * sqrt(numel(im(:,:,1))) ./ noise_per_band));
% SNR in
SNR_in(k) = 20*log10( norm(im(:)) / norm(im(:)-imnoise(:)));
% SNR out
SNR(k) = 20*log10( norm(im(:)) / norm(im(:)-imout(:)));
% SNR per spectral band
signal_per_band = sqrt(sum(sum(im.^2),2)); % noise norm per band
SNR_per_band(k,:) = squeeze(20*log10( signal_per_band ./ noise_per_band));


%% Saving results %%
params_light = rmfield(params,'x');

if (isfield(params,'lowrank') && params.lowrank)
    save(strcat('new_SNR_',num2str(params.dictsize),'atoms_algo',num2str(algo_type),'_sigma',num2str(round(sigma)),'_',imglist(imnum).name,'_trainnum',num2str(params.trainnum),'_',num2str(k_mc),'_gain',num2str(round(params.gain*1000)),'_lr',num2str(params.lowrank)),'params_light','alpha','PSNR','PSNR_per_band','SNR_in','SNR','SNR_per_band','SNR_perIt_lr','SNR_perIt_sparse','approx_rank','exec_times');
else
    save(strcat('new_SNR_',num2str(params.dictsize),'atoms_algo',num2str(algo_type),'_sigma',num2str(sigma),'_',imglist(imnum).name,'_trainnum',num2str(params.trainnum),'_',num2str(k_mc),'_gain',num2str(round(params.gain*1000))),'params_light','alpha','PSNR','PSNR_per_band','SNR_in','SNR','SNR_per_band','exec_times');
end

%Saving dictionary
if save_dict
    save(['dict_algo' num2str(algo_type) '_alpha' num2str(alpha(k))], 'dict','dict_struct','normVec')
end

end
end
end

%% Show results (complete experiment) %%
% Not done yet

end
