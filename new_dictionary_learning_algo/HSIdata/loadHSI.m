function im = loadHSI(imgname)

switch imgname
    case {'SanDiego','SanDiego-truncated','SanDiego-reshaped-2D', ...
          'SanDiego-small', 'SanDiego-reshaped-2D-small'}
        % San Diego
        Asd = load('SanDiego.mat'); % Asd.A original dimensions: 160000x158
        im  = uint16(reshape(Asd.A,400,400,158));
        
        % truncation similar to 
        %"Hyperspectral Image Denoising via Sparse
        %Representation and Low-Rank Constraint"
        % Actually, the spectral truncation cannot be reproduced.
        if strcmp(imgname,'SanDiego-truncated')
            im = im(end-255:end,end-255:end,21:120);
        end
        
        % truncation: very small image, for quick tests
        if strcmp(imgname,'SanDiego-small') || strcmp(imgname,'SanDiego-reshaped-2D-small')
            im = im(end-40:end,end-40:end,21:60); % very small
            if strcmp(imgname,'SanDiego-reshaped-2D-small')
                im  = reshape(im,41*41,40);
            end
        end
        
        % reshaped 2D image as done in
        %"Hyperspectral Image Denoising via Sparse
        %Representation and Low-Rank Constraint"        
        if strcmp(imgname,'SanDiego-reshaped-2D')
            im  = uint16(reshape(Asd.A,400*400,158));
        end

        
    case {'Washington','Washington-truncated','Washington-reshaped-2D', ...
          'Washington-small'}
        % Washington DC Mall (HyDICE Sensor)
        im = imread('dc.tif');

        % truncation similar to 
        %"Hyperspectral Image Denoising via Sparse
        %Representation and Low-Rank Constraint"
        % Actually, the spectral truncation cannot be reproduced.
        if strcmp(imgname,'Washington-truncated')
            im = im(end-345:end-90,1:256,:);   
        end
                
        % reshaped 2D image as done in
        %"Hyperspectral Image Denoising via Sparse
        %Representation and Low-Rank Constraint"        
        if strcmp(imgname,'Washington-reshaped-2D')
            im  = uint16(reshape(im,1280*307,191));
        end
    
                
        % truncation: very small image, for quick tests
        if strcmp(imgname,'Washington-small')
            im = im(end-40:end,end-40:end,21:60); % very small
        end
        
    case {'Urban', 'Urban-small'}
        
        Au = load('Urban.mat');
        im = reshape(Au.A,307,307,162);

        % truncation: very small image, for quick tests
        if strcmp(imgname,'Urban-small')
            im = im(end-40:end,end-40:end,21:60); % very small
        end
        
        % This signal ranges from 0 to 1000. Converting to [0, 256*128]
        im = uint16(im*(2^15)/1000);
        
    case {'Houston','Houston-truncated-left','Houston-truncated-center','Houston-small'}
        % Original size: 349x1905x144
        im = imread('Houston.tif');

        % truncation: 256x256x100 - left portion
        if strcmp(imgname,'Houston-truncated-left')
            im = im(50+(1:256),200 + (1:256),16:115); % spectral mode (taking central portion)
            % bands 1:10 and 120:130 have much lower energy and would
            % become too noisy for uniform noise 
            % Rasti2018 use a Gaussian noise profile along specral bands.
        end
        
        % truncation: 256x256x100 - central portion
        if strcmp(imgname,'Houston-truncated-center')
            % Middle
            im = im(50+(1:256),650 + (1:256),16:115); % spectral mode (taking central portion)
        end
        
        % truncation: very small image, for quick tests
        if strcmp(imgname,'Houston-small')
            im = im(50+(1:41),200 + (1:41),51:90); % very small
        end

        % This signal ranges [0, 256*256]. Converting to [0, 256*128]
        im = double(im)/2;
    otherwise
        error('Chosen Hyperspectral image is not available!')
end

% Rescaling pixels to range [0, 255]

% uint8 output
% im = uint8(im/(2^7)); % same as: im = uint8(double(im)*(2^8)/(2^15));

% Directly double - not exactly same result as above
% im = double(im)*(2^8-1)/(2^15);



end