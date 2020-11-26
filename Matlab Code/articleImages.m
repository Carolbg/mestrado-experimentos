%% Generate images for article - health

A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.1.S.2013-03-20.00.txt') ; 
f = figure;
% cmap = colormap(f,jet);
h = imagesc(A);
colorbar
saveas(gcf,'health_frontal_174.png')




A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.2.S.2013-03-20.00.txt');
f = figure;
% cmap = colormap(f,jet);
h = imagesc(A);
colorbar
saveas(gcf,'health_lateral_174.png')
% 
% A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.3.S.2013-03-20.00.txt');
% A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.4.S.2013-03-20.00.txt');
% A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.5.S.2013-03-20.00.txt');



%% Generate images for article - sick

A = load('../../Imagens_TXT_Estaticas_Balanceadas/1Doente/T0181.1.1.S.2013-08-16.00.txt') ; 
f = figure;
% cmap = colormap(f,jet);
h = imagesc(A);
colorbar
saveas(gcf,'sick_frontal_181.png')



A = load('../../Imagens_TXT_Estaticas_Balanceadas/1Doente/T0181.1.4.S.2013-08-16.00.txt');
f = figure;
% cmap = colormap(f,jet);
h = imagesc(A);
colorbar
saveas(gcf,'sick_lateral_181.png')
% 

%% 

fullPath = '../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.1.S.2013-03-20.00.txt';

img = load(fullPath);
pSaudaveis = img;

% Essa parte aqui que trata a conversao pra RBG, das linhas 455 ate a
% 468
f = figure;
cmap = colormap(f,jet);
h = imagesc(img);
Cdata = h.CData;
cmap = colormap;

% make it into a index image.
cmin = min(Cdata(:));
cmax = max(Cdata(:));
m = length(cmap);

index = fix((Cdata-cmin)/(cmax-cmin)*m)+1;
% Then to RGB
RGB = ind2rgb(index, cmap);

minSaudaveis = min(RGB(:));
maxSaudaveis = max(RGB(:));
pSaudaveisRGB = RGB;

figure;
imagesc(RGB);

folderSaudaveis = 'health_174_RGB_preprocessing.png';
saveas(gcf, folderSaudaveis)
% 
% figure;
% subplot(1,2,1)
% imagesc(RGB);
% 
% subplot(1,2,2)
% histogram(RGB);
% 
% folderSaudaveis = 'health_174_RGB_preprocessing.png';
% saveas(gcf, folderSaudaveis)


%%

A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.1.S.2013-03-20.00.txt') ; 
imgFiltered = medfilt2(A);

B = maxk(imgFiltered(:),100);
meanTop10 = mean(B);

B = mink(imgFiltered(:),100);
meanBottom10 = mean(B);

minMaxImg = (imgFiltered - meanBottom10)/(meanTop10-meanBottom10);
%    
%     min(minMaxImg(:))
%     max(minMaxImg(:))
% %     
minMaxImg(minMaxImg < 0) = 0;
minMaxImg(minMaxImg > 1) = 1;

minMaxImg3D(:,:,1) = minMaxImg;
minMaxImg3D(:,:,2) = minMaxImg;
minMaxImg3D(:,:,3) = minMaxImg;
    
f = figure;
% cmap = colormap(f,jet);
h = imagesc(minMaxImg3D);
saveas(gcf,'minMax_health_frontal_174.png')

A = load('../../Imagens_TXT_Estaticas_Balanceadas/1Doente/T0181.1.4.S.2013-08-16.00.txt');
imgFiltered = medfilt2(A);

B = maxk(imgFiltered(:),100);
meanTop10 = mean(B);

B = mink(imgFiltered(:),100);
meanBottom10 = mean(B);

minMaxImg = (imgFiltered - meanBottom10)/(meanTop10-meanBottom10);
%    
%     min(minMaxImg(:))
%     max(minMaxImg(:))
% %     
minMaxImg(minMaxImg < 0) = 0;
minMaxImg(minMaxImg > 1) = 1;

minMaxImg3D(:,:,1) = minMaxImg;
minMaxImg3D(:,:,2) = minMaxImg;
minMaxImg3D(:,:,3) = minMaxImg;
    
f = figure;
% cmap = colormap(f,jet);
h = imagesc(minMaxImg3D);
saveas(gcf,'minMax_sick_lateral_181.png')
% 