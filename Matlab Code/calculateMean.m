%% Leitura imagens

saudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_allData_menosValidacaoRuim_rgb_DA/0Saudavel/';
cd(saudaveis);
files=dir('*.npy');
cd('../../Experimentos github/Matlab Code')

sizeSaudaveis = size(files,1);
pSaudaveis = cell(sizeSaudaveis,1);
pSaudaveisFiltered = cell(sizeSaudaveis,1);
allImages = zeros(sizeSaudaveis*2, 480, 640, 3);

for i = 1:sizeSaudaveis
    fileName=files(i).name;

    fullPath = strcat(saudaveis, fileName);
    imgNp = py.numpy.load(fullPath);
    img = double(imgNp);
  
    allImages(i, :, :, :) = img;
    
end

doentes = '../../Imagens_TXT_Estaticas_Balanceadas_allData_menosValidacaoRuim_rgb_DA/1Doente/';
cd(doentes);
files=dir('*.npy');
cd('../../Experimentos github/Matlab Code')

sizeDoentes = size(files,1);
pDoentes = cell(sizeDoentes,1);
pDoentesFiltered = cell(sizeDoentes,1);

minDoentes = ones(sizeDoentes);
maxDoentes = ones(sizeDoentes);

for i = 1:sizeDoentes
    
    fileName=files(i).name;
    fullPath = strcat(doentes, fileName);
    
    imgNp = py.numpy.load(fullPath);
    img = double(imgNp);
  
    allImages(188+i, :, :, :) = img;
    
end


%%
allMean = mean(allImages(:))
allStd = std(allImages(:))

%%
d1 = allImages(:,:,:,1);
meanD1 = mean(d1, 'all')
stdD1 = std(d1(:))
d2 = allImages(:,:,:,2);
meanD2 = mean(d2, 'all')
stdD2 = std(d2(:))
d3 = allImages(:,:,:,3);
meanD3 = mean(d3, 'all')
stdD3 = std(d3(:))
