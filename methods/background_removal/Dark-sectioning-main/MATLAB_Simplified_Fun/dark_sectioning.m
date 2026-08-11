function [image1] = dark_sectioning(image0,emwavelength,NA,pixelsize,factor)

%% Read data
image0 = 255*(image0 - min(image0(:)))./(max(image0(:))-min(image0(:)));
[Nx0,Ny0,~] = size(image0);
[Nx,Ny,~] = size(image0);
if Ny>Nx
    image0(Nx+1:Ny,:,:)=0;
elseif Ny<Nx
    image0(:,Ny+1:Nx,:)=0;
end
[Nx,Ny,Nz] = size(image0);

%% Reconstruction parameters
background = 0; % 0-middle,1-severve
pad = 1;        %1-sysemtic,0-pad0
denoise = 0;    % Guassion denoise
thres = 50;     % Threshold to distinguish background and information
divide = 0.5;

%% Padding edge
pad_size = 40;
result_stack = zeros(Nx,Ny,Nz);
for jz = 1:Nz
    if pad ==1
        image(:,:,jz) = padarray(image0(:,:,jz),[floor(Nx/pad_size)+1,floor(Ny/pad_size)+1],'symmetric');
    else
        image(:,:,jz) = padarray(image0(:,:,jz),[floor(Nx/pad_size)+1,floor(Ny/pad_size)+1]);
    end
end

%% Basic parameters
[params.Nx,params.Ny,~] = size(image);
params.NA = NA;
params.emwavelength = emwavelength;
params.pixelsize = pixelsize;
params.factor = factor;

%% Background setting
if background == 1
    maxtime = 2;
    deg_matrix = [6,3,1.2];   % 3-10
    dep_matrix = [3,2,2];   % 0.7-2
    hl_matrix = [1,1,1];    % 3-8
elseif background == 0
    maxtime=1;
    deg_matrix = [6];   % 3-10
    dep_matrix = [3];   % 0.7-2
    hl_matrix = [1];    % 3-8
end

%% Dark sectioning
for time = 1:maxtime
    for jz = 1:Nz
        deg = deg_matrix(time);   % 3-10
        dep = dep_matrix(time);   % 0.7-2
        hl = hl_matrix(time);    % 3-8
        % Seperate spectrum and confirm block size
        [Hi,Lo,lp,EL] = separateHiLo(squeeze(image(:,:,jz)),params,deg,divide);
        block_size = confirm_block(params,lp);
        % Remove background for low-frequency part
        Lo_process = dehaze_fast2(Lo, 0.95, block_size, EL,dep,thres);
        result = Lo_process/hl + Hi;
        % Cutting edge
        result = result(floor(Nx/pad_size)+2:floor(Nx/pad_size)+Nx+1,floor(Ny/pad_size)+2:floor(Ny/pad_size)+Ny+1);
        % Saving results
        result_stack(:,:,jz) = result;
    end
    image0 = result_stack;
    for jz = 1:Nz
        if pad==1
            image(:,:,jz) = padarray(image0(:,:,jz),[floor(Nx/pad_size)+1,floor(Ny/pad_size)+1],'symmetric');
        else
            image(:,:,jz) = padarray(image0(:,:,jz),[floor(Nx/pad_size)+1,floor(Ny/pad_size)+1]);
        end
    end
end

%% Denoise
result_final = zeros(Nx,Ny,Nz);
for jz = 1:Nz
    if pad ==1
        temp = padarray(result_stack(:,:,jz),[floor(Nx/pad_size)+1,floor(Ny/pad_size)+1],'symmetric');
    else
        temp = padarray(result_stack(:,:,jz),[floor(Nx/pad_size)+1,floor(Ny/pad_size)+1]);
    end
    if denoise == 0
        temp1 = temp;
    else
        W = fspecial('gaussian',[2,2],1);
        temp1 = imfilter(temp, W, 'replicate');
    end
    result_final(:,:,jz) = temp1(floor(Nx/pad_size)+2:floor(Nx/pad_size)+Nx+1,floor(Ny/pad_size)+2:floor(Ny/pad_size)+Ny+1);
end

%% Select regin of interst
if Nx0~=Nx || Ny0~=Ny
    if Nx>Nx0
        image0(Nx0+1:Nx,:,:)=[];
    end
    if Ny0>Ny
        image0(:,Ny0+1:Ny,:)=[];
    end
end

%% Saving results
maxnum = max(max(max(result_final)));
image1 = 65535*result_final./maxnum;

end