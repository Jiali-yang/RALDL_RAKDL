function mydenoisedemo
% generate noisy image 

sigma_set = [15,25,50];
for i=1:3
    params=[];
sigma=sigma_set(i);
disp(' ');
disp('Generating noisy image...');

im1=imread('peppers_gray.jpg');
im = double(im1);

n = randn(size(im)) * sigma;
imnoise = im + n;

%imnoise=awgn(im,params.psnr); 

% set parameters 

params.x = imnoise;
params.blocksize = 16;
params.dictsize = 1024;
params.sigma = sigma;
params.maxval = 255;
params.stepsize=2;
params.trainnum = 30000;
params.iternum = 5;
params.memusage = 'high';


% denoise!

disp('Performing RAODL denoising...');
alg='RAODL';
t1=tic;
[imout, dict] = mydenoise(params,alg); 
total_t = toc(t1);
result=20*log10(params.maxval * sqrt(numel(im)) / norm(im(:)-imout(:)));
file=strcat(alg,'_',num2str(i), '.mat');
save(['exp2_Result\', file],'result','imout','dict','total_t');

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
% % 
