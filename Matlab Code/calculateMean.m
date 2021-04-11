%% Leitura imagens

saudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu_34each/0Saudavel/';
cd(saudaveis);
files=dir('*.npy');
cd('../../Experimentos github/Matlab Code')

sizeSaudaveis = size(files,1);
pSaudaveis = cell(sizeSaudaveis,1);
pSaudaveisFiltered = cell(sizeSaudaveis,1);

allImagesSaudaveis = zeros(sizeSaudaveis, 480, 640, 3);

for i = 1:sizeSaudaveis
    fileName=files(i).name;

    fullPath = strcat(saudaveis, fileName);
    imgNp = py.numpy.load(fullPath);
    img = double(imgNp);
  
    allImagesSaudaveis(i, :, :, :) = img;
    
end

doentes = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu_34each/1Doente/';
cd(doentes);
files=dir('*.npy');
cd('../../Experimentos github/Matlab Code')

sizeDoentes = size(files,1);
pDoentes = cell(sizeDoentes,1);
pDoentesFiltered = cell(sizeDoentes,1);


allImagesDoentes = zeros(sizeDoentes, 480, 640, 3);

minDoentes = ones(sizeDoentes);
maxDoentes = ones(sizeDoentes);

for i = 1:sizeDoentes
    
    fileName=files(i).name;
    fullPath = strcat(doentes, fileName);
    
    imgNp = py.numpy.load(fullPath);
    img = double(imgNp);
  
    allImagesDoentes(i, :, :, :) = img;
    
end

%%
sumSaudaveis = sum(allImagesSaudaveis,'all');
sumDoentes = sum(allImagesDoentes,'all');
allSum = sumDoentes+sumSaudaveis;

countSaudaveis = numel(allImagesSaudaveis);
countDoentes = numel(allImagesDoentes);
allCount = countSaudaveis+countDoentes;

meanCalculated = allSum/allCount

%%
allImagesSaudaveis = allImagesSaudaveis - meanCalculated;
allImagesSaudaveis = allImagesSaudaveis.^2;
sum1=sum(allImagesSaudaveis,'all');

allImagesDoentes = allImagesDoentes - meanCalculated;
allImagesDoentes = allImagesDoentes.^2;
sum2=sum(allImagesDoentes,'all');

sumPower =  sum1+ sum2; 

divisao = sumPower/allCount;
stdCalculated=sqrt(divisao)


%%
% allImages = [allImagesDoentes;allImagesSaudaveis];
% size(allImages)
% allMean = mean(allImages(:))
% allStd = std(allImages(:))

% %%
% d1 = allImages(:,:,:,1);
% meanD1 = mean(d1, 'all')
% stdD1 = std(d1(:))
% d2 = allImages(:,:,:,2);
% meanD2 = mean(d2, 'all')
% stdD2 = std(d2(:))
% d3 = allImages(:,:,:,3);
% meanD3 = mean(d3, 'all')
% stdD3 = std(d3(:))
