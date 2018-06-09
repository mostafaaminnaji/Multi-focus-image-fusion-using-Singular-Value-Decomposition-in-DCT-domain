% C)Mostafa Amin-Naji, Babol Noshirvani University of Technology,
% My Official Website: www.Amin-Naji.com
% My Email: Mostafa.Amin.Naji@Gmail.com

% PLEASE CITE THE BELOW PAPER IF YOU USE THIS CODE

% Mostafa Amin-Naji, Pardis Ranjbar-Noiey, Ali Aghagolzadeh, “Multi-focus 
% image fusion using Singular Value Decomposition in DCT domain,” in 2017 
% 10th Iranian Conference on Machine Vision and Image Processing (MVIP),
% 2017, pp. 45-51.
% DOI:  https://doi.org/10.1109/IranianMVIP.2017.8342367 

clc
clear
close all

%Select First Image
disp('Please Select First Image:')
[filename, pathname]= uigetfile({'*.jpg;*.png;*.tif'},'Select First Image');
path=fullfile(pathname, filename);
im1=imread(path);
disp('Great! First Image is selected')

%Select Second Image
disp('Please Select Second Image:')
[filename, pathname]= uigetfile({'*.jpg;*.png;*.tif'},'Select Second Image');
path=fullfile(pathname, filename);
im2=imread(path);
disp('Great! Second Image is selected')


if size(im1,3) == 3     % Check if the images are grayscale
    im1 = rgb2gray(im1);
end
if size(im2,3) == 3
    im2 = rgb2gray(im2);
end

if size(im1) ~= size(im2)	% Check if the input images are of the same size
    error('Size of the source images must be the same!')
end

disp('congratulations! Fusion Process in Running')

% Get input image size
[m,n] = size(im1);
FusedDCT = zeros(m,n);
FusedDCT_CV = zeros(m,n);
Map = zeros(floor(m/8),floor(n/8));	

% Level shifting
im1 = double(im1)-128;
im2 = double(im2)-128;

% Divide source images into 8*8 blocks and perform the fusion process
for i = 1:floor(m/8)
    for j = 1:floor(n/8)
        
        im1_Block = im1(8*i-7:8*i,8*j-7:8*j);
        im2_Block = im2(8*i-7:8*i,8*j-7:8*j);
        % Compute the 2-D DCT of 8*8 blocks 
        im1_Block_DCT = dct2(im1_Block);
        im2_Block_DCT = dct2(im2_Block);
        sigma1=svd(im1_Block_DCT);
        sigma2=svd(im2_Block_DCT);
        
        x1=sigma1(1)*sigma1(2)*sigma1(3)*sigma1(4)*sigma1(5);
        x2=sigma2(1)*sigma2(2)*sigma2(3)*sigma2(4)*sigma2(5);
        
         % Fusion Process
        if x1 > x2
            dctBlock = im1_Block_DCT;
            Map(i,j) =+1;	% Consistency verification 
        else
            dctBlock = im2_Block_DCT;
            Map(i,j) = -1;    % Consistency verification 
        end
        
        % Compute the 2-D inverse DCT of 8*8 blocks and construct fused image
        % DCT+SVD Method
        FusedDCT(8*i-7:8*i,8*j-7:8*j) = idct2(dctBlock);
        
    end
end

% Concistency verification (CV) with Majority Filter (3x3 Averaging Filter)

Filter=fspecial('average',3);

Map_Filtered = imfilter(Map, Filter,'symmetric');	

% The CV process
for i = 1:m/8
    for j = 1:n/8
        
        if Map_Filtered(i,j) > 0
            FusedDCT_CV(8*i-7:8*i,8*j-7:8*j) = im1(8*i-7:8*i,8*j-7:8*j);
        else
            FusedDCT_CV(8*i-7:8*i,8*j-7:8*j) = im2(8*i-7:8*i,8*j-7:8*j);
        end
        
    end
end

% Inverse level shifting 
im1 = uint8(double(im1)+128);
im2 = uint8(double(im2)+128);
FusedDCT = uint8(double(FusedDCT)+128);
FusedDCT_CV = uint8(double(FusedDCT_CV)+128);

% Show Images Table
subplot(2,2,1), imshow(im1), title('Source image 1');
subplot(2,2,2), imshow(im2), title('Source image 2');
subplot(2,2,3), imshow(FusedDCT), title('"DCT+SVD" fusion result');
subplot(2,2,4), imshow(FusedDCT_CV), title('"DCT+SVD+CV" fusion result');

% Good Luck
% Mostafa Amin-Naji ;)
