%% IMAGEM FRONTAL

A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0275.1.1.S.2015-03-13.00.txt') ; 
I = A;
figure;
imagesc(A)
%% Convert
I = A;
%I = medfilt2(I);

top = max(I(:));
I = uint8((255/top)*I);
imagesc(I)

[L,Centers] = imsegkmeans(I,2, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_2', 'png')
%saveas(fig, 'imsegkmeans_2_medianFilter', 'png')

[L,Centers] = imsegkmeans(I,3, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_3', 'png')
%saveas(fig, 'imsegkmeans_3_medianFilter', 'png')

[L,Centers] = imsegkmeans(I,4, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_4', 'png')
%saveas(fig, 'imsegkmeans_4_medianFilter', 'png')


[L,Centers] = imsegkmeans(I,5, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_5', 'png')
%saveas(fig, 'imsegkmeans_5_medianFilter', 'png')

%%
% r1 = 163;
% c1 = 37;
% figure
% contour = bwtraceboundary(I,[r1 c1],'W');
% hold on
% plot(contour(:,2),contour(:,1),'g','LineWidth',2)
% 
% figure
% B = bwboundaries(I);
% imshow(I)
% hold on
% visboundaries(B)


%%
I = A;
%I = medfilt2(I);
[~, threshold] = edge(I, 'canny');
BWs = edge(I,'canny', threshold);
fig = imagesc(BWs)
title('canny')
saveas(fig, 'canny', 'png')
%saveas(fig, 'canny_medianFilter', 'png')

se90 = strel('line', 3, 90); 
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
figure
fig = imshow(BWsdil);
title('dilated gradient mask');
saveas(fig, 'canny_dilatedGradient', 'png')
%saveas(fig, 'canny_dilatedGradient_medianFilter', 'png')

BWdfill = imfill(BWsdil, 'holes'); figure, 
fig = imshow(BWdfill);
title('binary image with filled holes');
saveas(fig, 'canny_filledHoles', 'png')
%saveas(fig, 'canny_filledHoles_medianFilter', 'png')

% BWnobord = imclearborder(BWdfill, 4);
% figure, imshow(BWnobord), title('cleared border image');

% seD = strel('diamond',1);
% BWfinal = imerode(BWnobord,seD);
% BWfinal = imerode(BWfinal,seD);
% figure, imshow(BWfinal), title('segmented image');


%%
[~, threshold] = edge(I, 'sobel');
BWs = edge(I,'sobel', threshold);
figure
fig = imagesc(BWs);
title('sobel')
saveas(fig, 'sobel', 'png')

[~, threshold] = edge(I, 'prewitt');
BWs = edge(I,'prewitt', threshold);
figure
fig = imagesc(BWs);
title('Prewitt')
saveas(fig, 'prewitt', 'png')

[~, threshold] = edge(I, 'roberts');
BWs = edge(I,'roberts', threshold);
figure
fig = imagesc(BWs);
title('roberts')
saveas(fig, 'roberts', 'png')

[~, threshold] = edge(I, 'log');
BWs = edge(I,'log', threshold);
figure
fig = imagesc(BWs);
title('log')
saveas(fig, 'log', 'png')

[~, threshold] = edge(I, 'zerocross');
BWs = edge(I,'zerocross', threshold);
figure
fig = imagesc(BWs);
title('zerocross')
saveas(fig, 'zerocross', 'png')
% 
% [~, threshold] = edge(I, 'approxcanny');
% BWs = edge(I,'approxcanny', threshold);
% figure
% fig = imagesc(BWs);
% title('approxcanny')
% saveas(fig, 'approxcanny', 'png')

%%

I = A;
%I = medfilt2(I);

%Nro de thresholds
figure
thresh = multithresh(I);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 2 classes')
saveas(fig, 'otsu_2_classes', 'png')
%saveas(fig, 'otsu_2_classes_medianFilter', 'png')

figure
thresh = multithresh(I,2);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 3 classes')
saveas(fig, 'otsu_3_classes', 'png')
%saveas(fig, 'otsu_3_classes_medianFilter', 'png')

figure
thresh = multithresh(I,3);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 4 classes')
saveas(fig, 'otsu_4_classes', 'png')
%saveas(fig, 'otsu_4_classes_medianFilter', 'png')

figure
thresh = multithresh(I,4);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 5 classes')
saveas(fig, 'otsu_5_classes', 'png')
%saveas(fig, 'otsu_5_classes_medianFilter', 'png')

%% 
% L = watershed(I);
% figure; imshow(L);
% 
% filtrada = medfilt2(I);
% L = watershed(filtrada);
% figure; imshow(L);
% title('1')
% BW(L == 0) = 0;
% figure; imshow(BW)
% title('2')

% 
% I2 = imtophat(I, strel('disk', 10));
% figure; imshow(I2)
% level = graythresh(I2);
% BW = im2bw(I2,level);
% figure; imshow(BW)
% 
% D = -bwdist(~BW);
% D(~BW) = -Inf;
% L = watershed(D);
% figure; imshow(label2rgb(L,'jet','w'))

% L = watershed(imcomplement(I));
% figure;
% imshow(L)

% I2 = imcomplement(I);
% I3 = imhmin(I2,20); %20 is the height threshold for suppressing shallow minima
% L = watershed(I3);
% imshow(L)

% gmag = imgradient(I);
% imshow(gmag,[])
% title('Gradient Magnitude')
% 
% se = strel('disk',20);
% Io = imopen(I,se);
% imshow(Io)
% title('Opening')
% 
% Ie = imerode(I,se);
% Iobr = imreconstruct(Ie,I);
% imshow(Iobr)
% title('Opening-by-Reconstruction')
% 
% 
% Iobrd = imdilate(Iobr,se);
% Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
% Iobrcbr = imcomplement(Iobrcbr);
% imshow(Iobrcbr)
% title('Opening-Closing by Reconstruction')
% 
% fgm = imregionalmax(Iobrcbr);
% imshow(fgm)
% title('Regional Maxima of Opening-Closing by Reconstruction')
% 
% I2 = labeloverlay(I,fgm);
% imshow(I2)
% title('Regional Maxima Superimposed on Original Image')


%% TESTANDO IMAGEM LATERAL

imgLateral = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0275.1.2.S.2015-03-13.00.txt') ; 
figure;
imagesc(imgLateral)
I = imgLateral;

top = max(I(:));
I = uint8((255/top)*I);
imagesc(I)

[L,Centers] = imsegkmeans(I,2, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_2_lateral', 'png')

[L,Centers] = imsegkmeans(I,3, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_3_lateral', 'png')

[L,Centers] = imsegkmeans(I,4, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_4_lateral', 'png')


[L,Centers] = imsegkmeans(I,5, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_5_lateral', 'png')


I = imgLateral;
%Nro de thresholds
figure
thresh = multithresh(I);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 2 classes')
saveas(fig, 'otsu_2_classes_lateral', 'png')

figure
thresh = multithresh(I,2);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 3 classes')
saveas(fig, 'otsu_3_classes_lateral', 'png')

figure
thresh = multithresh(I,3);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 4 classes')
saveas(fig, 'otsu_4_classes_lateral', 'png')

figure
thresh = multithresh(I,4);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 5 classes')
saveas(fig, 'otsu_5_classes_lateral', 'png')

I = imgLateral;
%I = medfilt2(I);
[~, threshold] = edge(I, 'canny');
BWs = edge(I,'canny', threshold);
fig = imagesc(BWs)
title('canny')
saveas(fig, 'canny_lateral', 'png')
%saveas(fig, 'canny_medianFilter', 'png')

se90 = strel('line', 3, 90); 
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
figure
fig = imshow(BWsdil);
title('dilated gradient mask');
saveas(fig, 'canny_dilatedGradient_lateral', 'png')
%saveas(fig, 'canny_dilatedGradient_medianFilter', 'png')

BWdfill = imfill(BWsdil, 'holes'); figure, 
fig = imshow(BWdfill);
title('binary image with filled holes');
saveas(fig, 'canny_filledHole_laterals', 'png')


%% Crop image frontal

croppedA = A(150:410, 90:550);
figure; imagesc(croppedA)

I = croppedA;

top = max(I(:));
I = uint8((255/top)*I);
imagesc(I)

[L,Centers] = imsegkmeans(I,2, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_2_cropped', 'png')

[L,Centers] = imsegkmeans(I,3, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_3_cropped', 'png')

[L,Centers] = imsegkmeans(I,4, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_4_cropped', 'png')


[L,Centers] = imsegkmeans(I,5, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_5_cropped', 'png')


I = croppedA;
%Nro de thresholds
figure
thresh = multithresh(I);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 2 classes')
saveas(fig, 'otsu_2_classes_cropped', 'png')

figure
thresh = multithresh(I,2);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 3 classes')
saveas(fig, 'otsu_3_classes_cropped', 'png')

figure
thresh = multithresh(I,3);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 4 classes')
saveas(fig, 'otsu_4_classes_cropped', 'png')

figure
thresh = multithresh(I,4);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 5 classes')
saveas(fig, 'otsu_5_classes_cropped', 'png')

I = croppedA;
%I = medfilt2(I);
[~, threshold] = edge(I, 'canny');
BWs = edge(I,'canny', threshold);
fig = imagesc(BWs)
title('canny')
saveas(fig, 'canny_cropped', 'png')
%saveas(fig, 'canny_medianFilter', 'png')

se90 = strel('line', 3, 90); 
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
figure
fig = imshow(BWsdil);
title('dilated gradient mask');
saveas(fig, 'canny_dilatedGradient_cropped', 'png')
%saveas(fig, 'canny_dilatedGradient_medianFilter', 'png')

BWdfill = imfill(BWsdil, 'holes'); figure, 
fig = imshow(BWdfill);
title('binary image with filled holes');
saveas(fig, 'canny_filledHole_cropped', 'png')



%%
fileName = strcat('teste_cortada.txt');
%imwrite(pSaudaveis{i}, strcat(saudaveis, imagesSaudaveis(i, :),'.png'));
fileID = fopen(fileName,'w');
%subplot(3,2,i)
[rows, columns, numberOfColorChannels] = size(seg_I)

imagesc(seg_I)
for j = 1:rows
    for k = 1:columns
        if k ~= 1
            fprintf(fileID, ' %.2f', seg_I(j,k));
        else
            fprintf(fileID, '%.2f', seg_I(j,k));
        end
    end
    if j < rows
        fprintf(fileID, '\n');
    end
end
