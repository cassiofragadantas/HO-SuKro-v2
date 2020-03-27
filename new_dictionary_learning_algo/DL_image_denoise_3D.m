function DL_image_denoise_3D
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

%% Downloading OMP and KSVD Toolboxes if necessary
FS=filesep;
pathstr = fileparts(which('DL_image_denoise_3D'));
addpath([pathstr,filesep,'..']); %addpath([pathstr,filesep,'..',filesep,'misc']);
addpath([pathstr,filesep,'..',filesep,'tensorlab_2016-03-28']);

SuKro_path = [pwd,FS,'toolbox/SuKro'];
addpath(SuKro_path);

KSVD_path = [pwd,FS,'toolbox'];

if ~exist('toolbox','dir')
    fprintf('\n ATTENTION: This is the first time you run this demonstration.');
    fprintf('\n Before starting the simulations, we will install some toolboxes.');
    fprintf('\n\n ****************** SETUP: Toolboxes installation ******************');
    fprintf('\n\n The following toolboxes will be downloaded:');
    fprintf('\n OMPbox version 10');
    fprintf('\n KSVDbox version 13');
    fprintf('\n\n IMPORTANT: To successfully install the toolboxes');
    fprintf('\n you will need to have MEX setup to compile C files.');
    fprintf('\n\n If this is not already setup, please type "n" to exit and then ');
    fprintf('\n run "mex -setup" or type "help mex" in the MATLAB command prompt.');
    fprintf('\n\n IMPORTANT: You must have an internet connection.');
    fprintf('\n\n ******************************************************************');

    install_ack = input('\n\n Do you wish to continue: (y/n)? ','s');

    if strcmp(install_ack,'"n"'),
      install_ack = 'n';
    end

    if install_ack == 'n',
      return;
    else
      fprintf('\n\n Installation now beginning...');

    end

    fprintf('\n ******************************************************************');
    fprintf('\n\n Initialising OMPbox and KSVDBox Setup');
    
    try
        if exist([KSVD_path, FS, 'ompbox10.zip'],'file'),
            omp_zip=[KSVD_path, FS, 'ompbox10.zip'];
        else
            omp_zip='http://www.cs.technion.ac.il/%7Eronrubin/Software/ompbox10.zip';
            fprintf('\n\n Downloading OMP toolbox, please be patient\n\n');
        end
        unzip(omp_zip,[KSVD_path, FS, 'ompbox']);
        
        cd([KSVD_path, FS, 'ompbox', FS, 'private']);
        make;
        cd(pathstr);
        
        if exist([KSVD_path, FS, 'ksvdbox13.zip'],'file'),
            KSVD_zip=[KSVD_path, FS, 'ksvdbox13.zip'];
        else
            KSVD_zip='http://www.cs.technion.ac.il/%7Eronrubin/Software/ksvdbox13.zip';
            fprintf('\n\n Downloading KSVD toolbox, please be patient\n\n');
        end
        unzip(KSVD_zip,[KSVD_path, FS, 'ksvdbox']);
        cd([KSVD_path, FS, 'ksvdbox', FS, 'private']);
        make;
        cd(pathstr);
        movefile('image_denoise.m',[KSVD_path, FS, 'ksvdbox']);
        fprintf('\n KSVDBox and OMPBox Installation Successful\n');
        fprintf('\n ******************************************************************');
        fprintf('\n\n >>> Now, please RERUN THIS SCRIPT to perform the demonstration <<< \n\n');
        return
    catch
        fprintf('\n KSVDBox and OMPBox Installation Failed');
        cd(pathstr);
    end
end
KSVD_p=genpath(KSVD_path);
addpath(KSVD_p);

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

dirname = fullfile(pathstr, 'images', '*.tiff');
imglist = dir(dirname);

disp('  Available test images:');
disp(' ');
for k = 1:length(imglist)
  fprintf('  %d. %s\n', k, imglist(k).name);
end
fprintf('  %d. All images', length(imglist)+1);
disp(' ');

imnum = 0;
while (~isnumeric(imnum) || (rem(imnum,1)~=0) || imnum<1 || imnum>length(imglist)+1)
  imnum = input(sprintf('  Image to denoise (%d-%d): ', 1, length(imglist)), 's');
  imnum = sscanf(imnum, '%d');
end

total_images = 1;
if imnum == length(imglist)+1, total_images = length(imglist), end

fprintf('\n\n  Choose the experiment type:\n');
fprintf('\n  1. Single-run (a few minutes)\n  2. Complete   (a few hours)\n');
exp_type = 0;
while (~isnumeric(exp_type) || (rem(exp_type,1)~=0) || exp_type<1 || exp_type>2)
  exp_type = input(sprintf('  Experiment type (1 or 2): '), 's');
  exp_type = sscanf(exp_type, '%d');
end

fprintf('\n\n  Choose the algorithm:\n');
fprintf('\n  1. SuKro\n  2. HO-SuKro\n  3. K-SVD\n  4. ODCT\n');
algo_type = 0;
while (~isnumeric(algo_type) || (rem(algo_type,1)~=0) || algo_type<1 || algo_type>4)
  algo_type = input(sprintf('  Algorithm type (1, 2, 3 or 4): '), 's');
  algo_type = sscanf(algo_type, '%d');
end
params.algo_type = algo_type;

for image = 1:total_images 

if image > 1, imnum = image; end
if imnum == 6, imnum = 1; end
imgname = fullfile(pathstr, 'images',imglist(imnum).name);

%% Simulation parameters %%
sigma = 75; fprintf('\nSIGMA = %d\n',sigma)
mc_it = 5; % Number of Monte-Carlo iterations

samples_training = 40000; %[1000 2000 5000 10000 20000]; % Number of training samples
params.iternum = 20;
params.blocksize = [12 12 3];
M3 = 6;
params.dictsize = 4*params.blocksize(1)*params.blocksize(2)*M3;
show_dict = false;
save_dict = false;

params.gain = 1.04;
params.sigma = sigma;
params.maxval = 255;
params.memusage = 'high';
params.my_omp_training = false; % MODIF: To use homemade OMP functions in denoising
if params.my_omp_training, fprintf('\n\n !!! \n USING HOMEMADE OMP DURING TRAINING\n !!!\n\n'); end
params.my_omp_denoise = false; % MODIF: To use homemade OMP functions in denoising
if params.my_omp_denoise, fprintf('\n\n !!! \n USING HOMEMADE OMP FOR DENOISING\n !!!\n\n'); end

if (exp_type == 1)  % Single-run experiment
    if algo_type == 1
        % SuKro
        alpha = 250;
    else
        % HO-SuKro
        alpha = 3;
    end
else                % Complete experiment
    if algo_type == 1
        alpha = logspace(log10(100),log10(600),50); % Nuclear norm regularization coefficient
    elseif algo_type == 2
        alpha = [1 2 3 4 5]; % Number of separable terms
        %alpha = 1:2:5; % Number of separable terms
        %alpha = [1 3]; % Number of separable terms
    else % KSVD or ODCT. This parameter is useless
        alpha = 1;
    end
end

im = imread(imgname);
im = double(im);

% Subdictionaries (Kronecker factors) dimensions
params.kro_dims.N1 = params.blocksize(1);
params.kro_dims.N2 = params.blocksize(2);
params.kro_dims.N3 = params.blocksize(3);
params.kro_dims.M1 = sqrt(params.dictsize/M3);
params.kro_dims.M2 = sqrt(params.dictsize/M3);
params.kro_dims.M3 = M3;

params.kro_dims.N = [params.kro_dims.N1 params.kro_dims.N2 params.kro_dims.N3];
params.kro_dims.M = [params.kro_dims.M1 params.kro_dims.M2 params.kro_dims.M3];

% Teste, denoising each color separately
%im = im(:,:,3); % TODO teste
%params.blocksize = [6 6];
%params.dictsize = 4*params.blocksize(1)*params.blocksize(2);

% results to store
SNR = zeros(size(alpha));
exec_times = cell(size(alpha));

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

[imout, dict, dict_struct, normVec, exec_times{k}] = image_denoise(params);

%% Show results (single-run experiment) %%
if (exp_type == 1)  % Single-run experiment
    % Show dictionary - only for 2-D
    if(show_dict) % Caution: This might be slow.
        cd toolbox/ksvdbox/private
        for k_atom=1:params.dictsize
            subplot(ceil(sqrt(params.dictsize)),ceil(sqrt(params.dictsize)),k_atom)
            imshow(imnormalize(reshape(dict(:,k_atom),params.blocksize)))
        end
        cd ../../../
    end
    figure; ax = subplot(1,1,1); imshow(im/params.maxval);
    title(ax,'Original image'); drawnow

    figure; ax = subplot(1,1,1); imshow(imnoise/params.maxval); 
    title(ax,sprintf('Noisy image, PSNR = %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imnoise(:))) ));

    figure; ax = subplot(1,1,1); imshow(imout/params.maxval);
    title(ax,sprintf('Denoised image, PSNR: %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:))) ));
end

SNR(k) = 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));

%% Saving results %%
params_light = rmfield(params,'x');
save(strcat('new_SNR_',num2str(params.dictsize),'atoms_algo',num2str(algo_type),'_sigma',num2str(sigma),'_',imglist(imnum).name(1:end-5),'_trainnum',num2str(params.trainnum),'_',num2str(k_mc)),'params_light','alpha','SNR','exec_times');
%Saving dictionary
if save_dict
    save(['dict_algo' num2str(algo_type) '_alpha' num2str(alpha(k))], 'dict','dict_struct','normVec')
end

end
end
end

%% Show results (complete experiment) %%
if (exp_type == 2)  % Complete experiment
    figure, hold on
    ylabel('Recovered Image PSNR [dB]');
    xlabel('Number of Separable Terms');
    title([imglist(imnum).name, '(input PSNR = ' num2str(10*log10(255^2/(sigma^2))) ')']);

    plot(alpha,SNR,'o');
    
    % ksvd and odct results
    pos = find([10 20  50 75 100] == sigma);
    if (pos) && (isequal(params.blocksize,[8 8])) && (params.dictsize == 256) && (params.trainnum == 40000)
        % Pre-calculated results for a specific parameter setting
        SNR_KSVD = [34.42 33.64 35.98 35.47 34.80 ;... % sigma 10
                    30.83 30.36 33.20 32.38 32.27 ;... % sigma 20
                    25.47 25.95 27.95 27.79 28.07 ;... % sigma 50
                    23.01 23.98 25.22 25.80 25.73 ;... % sigma 75
                    21.89 22.81 23.71 24.46 24.23];   % sigma 100 % From Elad-Aharon2006

        SNR_ODCT = [33.97 33.44 35.41 35.28 34.61 ;... % sigma 10
                    29.95 29.91 32.17 32.00 31.87 ;... % sigma 20
                    24.75 25.57 27.41 27.44 27.65 ;... % sigma 50
                    22.83 23.85 25.10 25.63 25.57 ;... % sigma 75
                    21.89 22.79 23.78 24.42 24.22];   % sigma 100

        plot([0 64], SNR_KSVD(pos,imnum)*ones(1,2),'k'); %ksvd
        plot([0 64], SNR_ODCT(pos,imnum)*ones(1,2),'k--'); %odct
    end
    grid on;
end

end
