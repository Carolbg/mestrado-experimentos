%% Leitura imagens

saudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/0Saudavel/';
cd(saudaveis);
files=dir('*.txt');
cd('../../Experimentos github/Matlab Code')

sizeSaudaveis = size(files,1);
pSaudaveis = cell(sizeSaudaveis,1);
pSaudaveisFiltered = cell(sizeSaudaveis,1);
allImages = zeros(sizeSaudaveis*2, 480, 640, 3);

for i = 1:sizeSaudaveis
    fileName=files(i).name;

    fullPath = strcat(nomeSaudaveis, fileName);
    
    img = load(fullPath);
    
    splittedFileName = split(fileName, '.txt');
    fileName = splittedFileName{1};

    RGBImg(:,:,1) = img;
    RGBImg(:,:,2) = img;
    RGBImg(:,:,3) = img;
    
    pSaudaveis{i} = RGBImg;
    
    imgFiltered = medfilt3(RGBImg, 'symmetric');
    minK = mink(imgFiltered(:),100);
    meanMin = mean(minK);
%     disp(['saudaveis i = ', num2str(meanMin)])
    
    pSaudaveisFiltered{i} = imgFiltered;
    allImages(i, :, :, :) = imgFiltered;
    
%     disp(['i = ', num2str(i), 'nomeSaudaveis(i, :)', nomeSaudaveis(i, :)])
end

doentes = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/1Doente/';
cd(doentes);
files=dir('*.txt');
cd('../../Experimentos github/Matlab Code')

sizeDoentes = size(files,1);
pDoentes = cell(sizeDoentes,1);
pDoentesFiltered = cell(sizeDoentes,1);

minDoentes = ones(sizeDoentes);
maxDoentes = ones(sizeDoentes);

for i = 1:sizeDoentes
    
    fileName=files(i).name;
    fullPath = strcat(nomeDoentes, fileName);
    
    img = load(fullPath); 
    
    RGBImg(:,:,1) = img;
    RGBImg(:,:,2) = img;
    RGBImg(:,:,3) = img;
    
    pDoentes{i} = RGBImg;
    
    imgFiltered = medfilt3(RGBImg, 'symmetric');
    
    minK = mink(imgFiltered(:),100);
    meanMin = mean(minK);
%     disp(['doentes i = ', num2str(meanMin)])
    
    pDoentesFiltered{i} = imgFiltered;
    allImages(188+i, :, :, :) = imgFiltered;
    
%     disp(['i = ', num2str(188+i)])
end

%% Calc min and max da base

maxBase = max(allImages(:))
minBase = min(allImages(:))

%% Aplica conversao 0-255

for i = 1:sizeSaudaveis
    imgFiltered = pSaudaveis{i};

    imgConverted = (255*(imgFiltered - meanBottom10))/(maxBase-minBase);
    imgConverted = round(imgConverted, 0);
    % disp(['< 0 = ', num2str(sum(minMaxImg(:) < 0)), ' e > 1 = ',num2str(sum(minMaxImg(:) >1))])
    % minMaxImg(imgConverted < 0) = 0;
    % minMaxImg(imgConverted > 1) = 1;
    

    %Saving original image
    numpyScaled = py.numpy.array(imgConverted);
    folderSaudaveis = strcat('../../Imagens_numpy_array_allData_entireDatabase_MinMax/0Saudaveis/', nomeSaudaveis(i, :));
    py.numpy.save(folderSaudaveis, numpyScaled);
    
%     dataAugment2DImage(imgConverted, nomeSaudaveis, i, 2, 'saudaveis/', '0Saudaveis','Imagens_numpy_array_allData_entireDatabase_MinMax')
    close all
end

for i = 1:sizeDoentes
    imgFiltered = pDoentes{i};

    imgConverted = (255*(imgFiltered - meanBottom10))/(maxBase-minBase);
    imgConverted = round(imgConverted, 0);
    
    %disp(['< 0 = ', num2str(sum(minMaxImg(:) < 0)), ' e > 1 = ',num2str(sum(minMaxImg(:) >1))])
    %minMaxImg(minMaxImg < 0) = 0;
    %minMaxImg(minMaxImg > 1) = 1;
    
    numpyScaled = py.numpy.array(imgConverted);
    folderDoentes = strcat('../../Imagens_numpy_array_allData_entireDatabase_MinMax_double/1Doentes/', nomeDoentes(i, :));
    py.numpy.save(folderDoentes, numpyScaled);
    
    dataAugment2DImage(imgConverted, nomeDoentes, i, 2, 'doentes/', '1Doentes', 'Imagens_numpy_array_allData_entireDatabase_MinMax_double')
    
    close all
end

