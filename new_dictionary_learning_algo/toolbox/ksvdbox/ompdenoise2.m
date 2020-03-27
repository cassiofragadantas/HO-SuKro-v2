function [y,nz] = ompdenoise2(params,msgdelta,D_struct,normVec)
%OMPDENOISE2 OMP denoising of 2-D signals.
%  OMPDENOISE2 denoises a 2-dimensional signal using OMP denoising. The
%  function syntax is identical to OMPDENOISE, but it runs significantly
%  faster on 2-D signals. OMPDENOISE2 requires somewhat more memory than
%  OMPDENOISE (approximately the size of the input signal), so if memory is
%  limited, OMPDENOISE can be used instead.
%
%  See also OMPDENOISE.


%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  August 2009


% parse input arguments %

x = params.x;
D = params.dict;
blocksize = params.blocksize;


% blocksize %
if (numel(blocksize)==1)
  blocksize = ones(1,2)*blocksize;
end


% maxval %
if (isfield(params,'maxval'))
  maxval = params.maxval;
else
  maxval = 1;
end


% gain %
if (isfield(params,'gain'))
  gain = params.gain;
else
  gain = 1.15;
end


% maxatoms %
if (isfield(params,'maxatoms'))
  maxatoms = params.maxatoms;
else
  maxatoms = floor(prod(blocksize)/2);
end


% stepsize %
if (isfield(params,'stepsize'))
  stepsize = params.stepsize;
  if (numel(stepsize)==1)
    stepsize = ones(1,2)*stepsize;
  end
else
  stepsize = ones(1,2);
end
if (any(stepsize<1))
  error('Invalid step size.');
end


% noise mode %
if (isfield(params,'noisemode'))
  switch lower(params.noisemode)
    case 'psnr'
      sigma = maxval / 10^(params.psnr/20);
    case 'sigma'
      sigma = params.sigma;
    otherwise
      error('Invalid noise mode specified');
  end
elseif (isfield(params,'sigma'))
  sigma = params.sigma;
elseif (isfield(params,'psnr'))
  sigma = maxval / 10^(params.psnr/20);
else
  error('Noise strength not specified');
end


% lambda %
if (isfield(params,'lambda'))
  lambda = params.lambda;
else
  lambda = maxval/(10*sigma);
end


% msgdelta %
if (nargin <2)
  msgdelta = 5;
end
if (msgdelta<=0)
  msgdelta = -1;
end


epsilon = sqrt(prod(blocksize)) * sigma * gain;   % target error for omp


MEM_LOW = 1;
MEM_NORMAL = 2;
MEM_HIGH = 3;

if (isfield(params,'memusage'))
  switch lower(params.memusage)
    case 'low'
      memusage = MEM_LOW;
    case 'normal'
      memusage = MEM_NORMAL;
    case 'high'
      memusage = MEM_HIGH;
    otherwise
      error('Invalid memory usage mode');
  end
else
  memusage = MEM_NORMAL;
end


% compute G %

G = [];
if (memusage >= MEM_NORMAL)
  G = D'*D;
end


% verify dictionary normalization %

if (isempty(G))
  atomnorms = sum(D.*D);
else
  atomnorms = diag(G);
end
if (any(abs(atomnorms-1) > 1e-2))
  error('Dictionary columns must be normalized to unit length');
end


% denoise the signal %


nz = 0;  % count non-zeros in block representations

% the denoised signal
y = zeros(size(x));

blocknum = prod(floor((size(x)-blocksize)./stepsize) + 1);
processedblocks = 0;
tid = timerinit('ompdenoise', blocknum);

for j = 1:stepsize(2):size(y,2)-blocksize(2)+1

  % the current batch of blocks
  blocks = im2colstep(x(:,j:j+blocksize(2)-1),blocksize,stepsize);

  % remove DC
  [blocks, dc] = remove_dc(blocks,'columns');

  % denoise the blocks
  
  % MY MODIFS
  if isfield(params, 'my_omp_denoise') && params.my_omp_denoise
      % Using my homemade OMP functions for fair comparison (Cassio)
      % Gives same result as the original implementation below. Verified!
      cleanblocks = zeros(size(blocks));
      for k_block = 1:size(blocks,2)
          if params.algo_type == 2 % Structure OMP
              [~, cleanblocks(:,k_block),nnz_gamma,~] = SolveOMP_2D(D, blocks(:,k_block), params.dictsize, D_struct,normVec.',params.blocksize,params.alpha, prod(params.blocksize),0,0,0,epsilon);
          else
              [~, cleanblocks(:,k_block),nnz_gamma,~] = SolveOMP(D, blocks(:,k_block), params.dictsize, prod(params.blocksize),0,0,0,epsilon);
          end
          nz = nz + nnz_gamma;
      end
      cleanblocks = add_dc(cleanblocks, dc, 'columns');
  else
      % This is the original implementation
      if (memusage == MEM_LOW)
        gamma = omp2(D,blocks,[],epsilon,'maxatoms',maxatoms,'checkdict','off');
      else
        gamma = omp2(D'*blocks,sum(blocks.*blocks),G,epsilon,'maxatoms',maxatoms,'checkdict','off');
      end
      nz = nz + nnz(gamma);
      cleanblocks = add_dc(D*gamma, dc, 'columns');
  end
  % END MY MODIFS
  
  cleanim = col2imstep(cleanblocks,[size(y,1) blocksize(2)],blocksize,stepsize);
  y(:,j:j+blocksize(2)-1) = y(:,j:j+blocksize(2)-1) + cleanim;

  if (msgdelta>0)
    processedblocks = processedblocks + size(blocks,2);
    timereta(tid, processedblocks, msgdelta);
  end

end

if (msgdelta>0)
  timereta(tid, blocknum);
end


nz = nz/blocknum;  % average number of non-zeros

% average the denoised and noisy signals
cnt = countcover(size(x),blocksize,stepsize);
%% Testing values of lambda and lambda0
% % Alg1 : calibrating lambda
% SNR = [];
% % lambda_vec =  [0 logspace(log10(lambda/10),log10(lambda*1000), 100)];
% lambda_vec =  [0 logspace(log10(lambda/10000),log10(lambda*1000), 10000)];
% for k_lambda = lambda_vec
%     yout = (y+k_lambda*x)./(cnt + k_lambda);             % Alg. 1
%     SNR = [SNR 20*log10( norm(params.im(:)) / norm(params.im(:)-yout(:)))];
% end
% [~, idx] = max(SNR);
% 
% % Uncomment this line for oracle version
% lambda = lambda_vec(idx); disp('  !!!! Using oracle lambda (averaging coef) !!!!')

y = (y+lambda*x)./(cnt + lambda);

