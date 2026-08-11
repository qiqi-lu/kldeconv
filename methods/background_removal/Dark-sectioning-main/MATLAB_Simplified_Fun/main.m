clear all;close all;clc;
image0 = double(imstackread('Mousekidney_561nm_1.49NA_65nm.tif'));
image1 = dark_sectioning(image0,610,1.49,65,2);
% 610: emission wavelength(nm)
% 1.49: NA
% 65: pixelsize(nm)
% 2: resolution factor
image0 = image0/max(image0(:));
image1 = image1/max(image1(:));
figure;imshow(squeeze(image0(:,256,:)));
figure;imshow(squeeze(image1(:,256,:)));