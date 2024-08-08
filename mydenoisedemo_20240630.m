%=======================================================================================%
% Image denoising(Stockton, Richmond, Shreveport,Oakland, wash-ir)
% Table 6 New experiment: add new high-definition images, all color images, grayscale processing
% Stockton2.2.20.jpg(1024)-2466_1;wash-ir.tiff(800)-2466_3;
% Richmond, Ca.2.2.04.tiff(1024)-2466_4;Shreveport2.2.14.tiff(1024)-2466_5;Oakland2.2.07.tiff(1024)-2466_6
%=======================================================================================%
function mydenoisedemo_20240630
addpath('.\packages\ksvdbox13')
addpath('.\standard_test_images')
% generate noisy image 
sigma_set = [15,25,50];
for i=4:4
    params=[];
sigma=sigma_set(i);
disp(' ');
disp('Generating noisy image...');

%Oakland2.2.07.tiff  Richmond, Ca.2.2.04.tiff  Shreveport2.2.14.tiff
%Stockton2.2.20.jpg  wash-ir.jpg
picture_1 = {'Stockton'};  % {'Oakland', 'Richmond', 'Shreveport'}  {'wash-ir'}

for pi=1:1
name = picture_1{pi};
pi_path = strcat('standard_test_images\', name, '.jpg');
%pi_path = strcat('standard_test_images\', name, '.tiff');
im1=imread(pi_path);
im1 = rgb2gray(im1);
%im1 = im2gray(im1);
im = double(im1);

n = randn(size(im)) * sigma;
imnoise = im + n;

%imnoise=awgn(im,params.psnr); 

% set parameters 

params.x = imnoise;
params.blocksize = 32;   
params.dictsize = 3000;   
params.trainnum = 40000;   
params.stepsize=3;     %with a step size of 4 for the Stockton, Richmond, Shreveport,Oakland images, and with a step size of 3 for the wash-ir image

params.sigma = sigma;
%params.noisemode = 'psnr';
params.maxval = 255;
params.iternum = 5;
params.memusage = 'high';


% denoise!
disp('Performing AKSVD denoising...');
alg='AKSVD';
t1=tic;
[imout, dict,compuD_t,compuX_t] = mydenoise(params,alg); 
total_t = toc(t1);
result=20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
file=strcat(num2str(name),alg,'_',num2str(i), '.mat');
save(['D:\code\Master\myresult2\denoise\test3_20240630\',file],'result','imout','dict','imnoise','compuD_t','compuX_t','total_t');
% 
disp('Performing KSVD denoising...');
alg='KSVD';
t1=tic;
[imout, dict,compuD_t,compuX_t] = mydenoise(params,alg);  
total_t = toc(t1);
result=20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
file=strcat(num2str(name),alg,'_',num2str(i), '.mat');
save(['D:\code\Master\myresult2\denoise\test3_20240630\',file],'result','imout','dict','compuD_t','compuX_t','total_t');
% % % 
disp('Performing MOD denoising...');
alg='MOD';
t1=tic;
[imout, dict,compuD_t,compuX_t] = mydenoise(params,alg);   
total_t = toc(t1);
result=20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
file=strcat(num2str(name),alg,'_',num2str(i), '.mat');
save(['D:\code\Master\myresult2\denoise\test3_20240630\',file],'result','imout','dict','compuD_t','compuX_t','total_t');
% 
disp('Performing SGK denoising...');
alg='SGK';
t1=tic;
[imout, dict,compuD_t,compuX_t] = mydenoise(params,alg);   
total_t = toc(t1);
result=20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
file=strcat(num2str(name),alg,'_',num2str(i), '.mat');
save(['D:\code\Master\myresult2\denoise\test3_20240630\',file],'result','imout','dict','compuD_t','compuX_t','total_t');

disp('Performing RFODL denoising...');
alg='RFODL';
t1=tic;
[imout, dict,compuD_t,compuX_t] = mydenoise(params,alg); 
total_t = toc(t1);
result=20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
file=strcat(num2str(name),alg,'_',num2str(i), '.mat');
save(['D:\code\Master\myresult2\denoise\test3_20240630\',file],'result','imout','dict','compuD_t','compuX_t','total_t');

end
end


% show results 
% 
% dictimg = showdict(dict,[1 1]*params.blocksize,round(sqrt(params.dictsize)),round(sqrt(params.dictsize)),'lines','highcontrast');
% figure; imshow(imresize(dictimg,2,'nearest'));
% title('Trained dictionary');

% figure; imshow(im/params.maxval); 
% title('Original image');
% 
% figure; imshow(imnoise/params.maxval); 
% title(sprintf('Noisy image, PSNR = %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imnoise(:))) ));

% figure; imshow(imout/params.maxval);
% title(sprintf('Denoised image, PSNR: %.2fdB', 20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:))) ));
% % S