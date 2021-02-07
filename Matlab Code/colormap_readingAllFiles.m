%% SCRIPT 1: 
% allSaudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/allSaudaveis/';
% cd(allSaudaveis);
% files=dir('*.txt');
% cd('../../Experimentos github/Matlab Code')
% for k=1:length(files)
%     fileName=files(k).name;
%    
%     fullPath = strcat(allSaudaveis, fileName);
%     img = load(fullPath);
%     f = figure;
%     imagesc(img)
%     title(fileName)
% 
% end


%% Leitura imagens saudaveis

nomeSaudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_allData_asCabıoglu/0Saudavel/';
cd(nomeSaudaveis);
files=dir('*.txt');
cd('../../Experimentos github/Matlab Code')

sizeSaudaveis = size(files,1);
pSaudaveis = cell(sizeSaudaveis,1);
pSaudaveisRGB = cell(sizeSaudaveis,1);
minSaudaveis = ones(sizeSaudaveis);
maxSaudaveis = ones(sizeSaudaveis);

for i = 1:sizeSaudaveis
    fileName=files(i).name;

    fullPath = strcat(nomeSaudaveis, fileName);
    
    img = load(fullPath);
    
    splittedFileName = split(fileName, '.txt');
    fileName = splittedFileName{1};
    
    pSaudaveis{i} = img;
    
    % Essa parte aqui que trata a conversao pra RBG
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
    
    minSaudaveis(i) = min(RGB(:));
    maxSaudaveis(i) = max(RGB(:));
    pSaudaveisRGB{i} = RGB;
    
    figure;
    subplot(1,2,1)
    imagesc(RGB);
    title(fileName)

    subplot(1,2,2)
    histogram(RGB);
  
    folderSaudaveis = strcat('imagesVerify/', fileName, '.png');
    saveas(gcf, folderSaudaveis)
    
    %Saving original image
    numpyRGB = py.numpy.array(RGB);
    folderSaudaveis = strcat('../../Imagens_TXT_Estaticas_Balanceadas_allData_asCabıoglu_DA/0Saudavel/', fileName);
    py.numpy.save(folderSaudaveis, numpyRGB);
%     
     dataAugment(img, RGB, fileName, i, 1, 'imagesVerify/', '0Saudavel','Imagens_TXT_Estaticas_Balanceadas_allData_asCabıoglu_DA')

    close all
end

%% Leitura imagens doentes


nomeDoentes = '../../Imagens_TXT_Estaticas_Balanceadas_allData_asCabıoglu/1Doente/';
cd(nomeDoentes);
files=dir('*.txt');
cd('../../Experimentos github/Matlab Code')

sizeDoentes = size(files,1);
pDoentes = cell(sizeDoentes,1);
pDoentesRGB = cell(sizeDoentes,1);

minDoentes = ones(sizeDoentes);
maxDoentes = ones(sizeDoentes);

for i = 1:sizeDoentes
    fileName=files(i).name;
    fullPath = strcat(nomeDoentes, fileName);
    
    img = load(fullPath);
    pDoentes{i} = img;
    
    splittedFileName = split(fileName, '.txt');
    fileName = splittedFileName{1};
    
     % 
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
    
    minDoentes(i) = min(RGB(:));
    maxDoentes(i) = max(RGB(:));
    pDoentesRGB{i} = RGB;
   
    figure;
    subplot(1,2,1)
    imagesc(RGB);
    title(fileName)

    subplot(1,2,2)
    histogram(RGB);
    
    folderDoentes = strcat('imagesVerify/', fileName, '.png');
    saveas(gcf, folderDoentes)
    
    numpyRGB = py.numpy.array(RGB);
    folderDoentes = strcat('../../Imagens_TXT_Estaticas_Balanceadas_allData_asCabıoglu_DA/1Doente/', fileName);
    py.numpy.save(folderDoentes, numpyRGB);
    
    dataAugment(img, RGB, fileName, i, 7, 'imagesVerify/', '1Doente', 'Imagens_TXT_Estaticas_Balanceadas_allData_asCabıoglu_DA')
    
    close all
end

