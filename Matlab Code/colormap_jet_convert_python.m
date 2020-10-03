%% Script para testes apenas e alinhamento antes de rodar pra todos os dados

%% a
A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0275.1.1.S.2015-03-13.00.txt') ; 
I = A;

%% 
f = figure;
cmap = colormap(f,jet);
h = imagesc(A);
colorbar

%%
% 
% F = getimage;
% dadosFrame = F.cdata;
    
Cdata = h.CData;
cmap = colormap;

% make it into a index image.
cmin = min(Cdata(:));
cmax = max(Cdata(:));
m = length(cmap);

index = fix((Cdata-cmin)/(cmax-cmin)*m)+1; %A
% Then to RGB
RGB = ind2rgb(index,cmap);

figure;
subplot(1,2,1)
cmap = colormap(f,jet);
imagesc(RGB);
title('test')

subplot(1,2,2)
histogram(RGB);
% saveas(gcf,'testing.png')
% numpyRGB = py.numpy.array(RGB);
% py.numpy.save('numpyTesting', numpyRGB);

%No python: x = np.load('numpyTesting.npy')

%% utilizando imageDataAugmenter
I = A;
RGBParsed = RGB;
thresh = multithresh(I);
seg_I = imquantize(I,thresh);
figure;imagesc(seg_I);
seg_I(seg_I == 2) = 0;
figure;imagesc(seg_I);
imgComMask = RGBParsed.* seg_I;
figure;imagesc(imgComMask)
meanValue1 = mean(nonzeros(imgComMask(:,:,1)));
meanValue2 = mean(nonzeros(imgComMask(:,:,2)));
meanValue3 = mean(nonzeros(imgComMask(:,:,3)));

augmenter = imageDataAugmenter('FillValue', [meanValue1, meanValue2, meanValue3],...
    'RandRotation',[0 360], ...
    'RandXReflection',true, ...
    'RandYReflection', true, ...
    'RandRotation',[-45, 45] ...
);
outCellArray = augment(augmenter,{RGB});
outImg = imtile(outCellArray);
figure; imshow(outImg);

%% 
imJittered = jitterColorHSV(RGB,'Saturation',[-0.4 -0.1]); 
montage({RGB,imJittered})


%% 
% top = mean(maxk(I(:),100));
% I = uint8((255/top)*I);
% imagesc(I)

% [L,Centers] = imsegkmeans(I,2, 'NormalizeInput', true);
% B = labeloverlay(I,L);
% fig = figure;
% imshow(B)
% 
% level=graythresh(I);
% BW = imbinarize(I,level);

I = A;
%I = medfilt2(I);

%Nro de thresholds
%%
I = A;
top = mean(maxk(I(:),100));
I = uint8((255/top)*I);
imagesc(I)
% top = mean(maxk(RGB(:),100));
% I= uint8((255/top)*RGB);
% imagesc(I)

[L,Centers] = imsegkmeans(I,2, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
I1 = B;
values = I1(L == 1);
meanValue1 = uint8(mean(values));
% values = I2(L == 1);
% meanValue2 = uint8(mean(values));
% values = I3(L == 1);
% meanValue3 = uint8(mean(values));

% values = I(L == 1);
% meanValue = uint8(mean(values));
% meanValue1 = mean(nonzeros(imgComMask(:,:,1)));
% meanValue2 = mean(nonzeros(imgComMask(:,:,2)));
% meanValue3 = mean(nonzeros(imgComMask(:,:,3)));

%%

I = A;
RGBParsed = RGB;
thresh = multithresh(I);
seg_I = imquantize(I,thresh);
figure;imagesc(seg_I);
RGBParsed1 = RGBParsed(:,:,1);
RGBParsed2 = RGBParsed(:,:,2);
RGBParsed3 = RGBParsed(:,:,3);
meanValue1 = mean(RGBParsed1(seg_I == 1));
meanValue2 = mean(RGBParsed2(seg_I == 1));
meanValue3 = mean(RGBParsed3(seg_I == 1));

% teste = RGBParsed(seg_I == 2);
% figure;imagesc(seg_I);
% imgComMask = RGBParsed.* seg_I;
% figure;imagesc(imgComMask)
% meanValue1 = mean(nonzeros(imgComMask(:,:,1)));
% meanValue2 = mean(nonzeros(imgComMask(:,:,2)));
% meanValue3 = mean(nonzeros(imgComMask(:,:,3)));
%%

imOriginal = RGB;
tform = randomAffine2d('Rotation',[-45 45], 'XReflection',true,'YReflection',true); 
outputView = affineOutputView(size(imOriginal),tform);
imAlterada = imwarp(imOriginal,tform,'OutputView',outputView,'FillValues',[meanValue1 meanValue2 meanValue3]);
figure
imagesc(imAlterada)
title('sem cores')

% imAlteradaCor = jitterColorHSV(imAlterada,'Contrast',[1.2 1.4],'Saturation',[-0.4 -0.1],'Brightness',[-0.2 0.2]);

% figure;
% imagesc(imAlterada)
% title('com cores')