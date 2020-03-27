    % Calls the DL_image_denoise_3D_input or DL_HSI_denoise_input function 
% for a given set of parameters.
%
% DL_image_denoise_3D_input(imnum, exp_type, algo_type, sigma, mc_it,samples_training,blocksize,blocksize_m)
% Exemple of call:
% DL_image_denoise_3D_input(2,2,2,20,5,[2000,40000],[6,6,3],[12,12,6])

data_type = 2;             % 1: Color image, 2: Hyperspectral Image

imnum = [1];                % For color images: 1 to 5 chooses one given image 
                            %                   (tree_color, tiffany_color, oakland_color, mandrill_color, lena_color).
                            %                   6: all 5 images.
                            % For HSI denoise (see HSIdata/loadHSI.m):
                            %                   imnum=1 : (SanDiego)
                            %                   2 : (SanDiego-truncated)
                            %                   6 : (Washington)
                            %                   10: (Urban)
                            %                   12: (Houston)
                            %                   13: (Houston-truncated-left)
                            
exp_type = 1;               %1. Single-run (a few minutes)  2. Complete   (a few hours)
algo_type = [2];              % 1. SuKro  2. HO-SuKro  3. K-SVD  4. ODCT
iternum = 20;               % Nb. iterations of Dict.update + Sparse coding. Default is 20.
sigma = [513];   % Noise standard deviation (on pixel values)
                                % For HSI denoise (SNR = [5 10 15 20 25 30]):
                                % imnum=1 : [1622 912 513 288 162 91.2]  (SanDiego)
                                % 2 : [1671 940 530 295 167 94]          (SanDiego-truncated)
                                % 6: [1703 957 539 303 170 95.7]         (Washington)
                                % 10: [3746 2107 1185 666 375 210.7]       (Urban)
                                % 12 : [1946 1095 615 346 194.5 109.5]   (Houston)
                                % 13 : [2934 1650 928 522 293 165]       (Houston-truncated-left)
mc_it = 10;                  % Number of noise realization iterations
samples_training = 20000;   % Number of training samples
                            % Color images (or HSI whispers): [1000 2000 5000 10000 20000];
                            % HSI: 100000;
% n = [6, 6, 6];                % 3D-data dimensions 
% m = [12, 12, 12];              % nb. atoms of each subdictionary D1, D2 and D3 respectively
n = [6, 6];                % 2D-data dimensions (used in Whisper's paper)
m = [12, 12];              % nb. atoms of each subdictionary D1, D2 and D3 respectively

gain = [1.16]; %1:00.01:1.10; % Calibrated gains:
                            % Color image: gain=1.04 (with n = [6, 6, 3]),
                            %              gain=1.02 (with n = [12, 12, 6])
                            % HSI: gain_KSVD=1.06, gain_SuKro=1.02
                            % HSI When using params.lowrank = true
                            % (i.e. Whispers paper): 1.16

for k_algo_type = algo_type
    for k_sigma = sigma
        for k_imnum = imnum
            for k_gain = gain
                close all
                if data_type == 1
                    DL_image_denoise_3D_input(k_imnum, exp_type, k_algo_type, iternum, k_sigma, mc_it,samples_training,n,m,k_gain);
                else
                    DL_HSI_denoise_input(k_imnum, exp_type, k_algo_type, iternum, k_sigma, mc_it,samples_training,n,m,k_gain);
                end
            end
        end
    end
end
