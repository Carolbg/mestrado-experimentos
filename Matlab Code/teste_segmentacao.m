%%

A = load('../../testes_segmentacao/T0275.1.1.S.2015-03-13.00.txt') ; 
I = A;
figure;
imagesc(A)
%% Convert
top = max(I(:));
I = uint8((255/top)*I);
imagesc(I)

[L,Centers] = imsegkmeans(I,2, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_2', 'png')

[L,Centers] = imsegkmeans(I,3, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_3', 'png')

[L,Centers] = imsegkmeans(I,4, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_4', 'png')


[L,Centers] = imsegkmeans(I,5, 'NormalizeInput', true);
B = labeloverlay(I,L);
fig = figure;
imshow(B)
saveas(fig, 'imsegkmeans_5', 'png')

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

%% VER MELHOR
L = watershed(A, 4);
imshow(L)

%%
I = medfilt2(I);
[~, threshold] = edge(I, 'canny');
BWs = edge(I,'canny', threshold);
fig = imagesc(BWs)
title('canny')
%saveas(fig, 'canny', 'png')
saveas(fig, 'canny_medianFilter', 'png')

se90 = strel('line', 3, 90); 
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
figure
fig = imshow(BWsdil);
title('dilated gradient mask');
%saveas(fig, 'canny_dilatedGradient', 'png')
saveas(fig, 'canny_dilatedGradient_medianFilter', 'png')

BWdfill = imfill(BWsdil, 'holes'); figure, 
fig = imshow(BWdfill);
title('binary image with filled holes');
%saveas(fig, 'canny_filledHoles', 'png')
saveas(fig, 'canny_filledHoles_medianFilter', 'png')

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

[~, threshold] = edge(I, 'approxcanny');
BWs = edge(I,'approxcanny', threshold);
figure
fig = imagesc(BWs);
title('approxcanny')
saveas(fig, 'approxcanny', 'png')

%%

I = A;
%Nro de thresholds
figure
thresh = multithresh(I);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 2 classes')
saveas(fig, 'otsu_2_classes', 'png')

figure
thresh = multithresh(I,2);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 3 classes')
saveas(fig, 'otsu_3_classes', 'png')

figure
thresh = multithresh(I,3);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 4 classes')
saveas(fig, 'otsu_4_classes', 'png')

figure
thresh = multithresh(I,4);
seg_I = imquantize(I,thresh);
fig = imagesc(seg_I);
title('otsu 5 classes')
saveas(fig, 'otsu_5_classes', 'png')



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
