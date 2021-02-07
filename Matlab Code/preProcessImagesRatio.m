function [pSaudaveisRatio, pSaudaveisAugmentedRatio, pDoentesRatio, pDoentesAugmentedRatio] = preProcessImagesRatio(hasDataAugment, originFolderName, toFolderName, numberAugmentedHealthImages, numberAugmentedSickImages)

%Leitura imagens saudaveis

    folderClassSaudaveis = '0Saudavel';
    nomeSaudaveis = strcat('../../', originFolderName, '/', folderClassSaudaveis);
    cd(nomeSaudaveis);
    filesSaudaveis = dir('*.txt');
    cd('../../Experimentos github/Matlab Code')
    
    folderClassDoentes = '1Doente';
    nomeDoentes = strcat('../../',originFolderName,'/', folderClassDoentes);
    cd(nomeDoentes);
    filesDoentes = dir('*.txt');
    cd('../../Experimentos github/Matlab Code')
    
    %Reading the data
    [saudaveisData, pSaudaveis, allSaudaveisFileNames] = readImages(filesSaudaveis, nomeSaudaveis);
    [doentesData, pDoentes, allDoentesFileNames] = readImages(filesDoentes, nomeDoentes);
    
    if hasDataAugment
        [augmentedSaudaveis, pAugmentedSaudaveis] = applyDA(pSaudaveis, numberAugmentedHealthImages, filesSaudaveis);
        allSaudaveis = [saudaveisData; augmentedSaudaveis];
    
        [augmentedDoentes, pAugmentedDoentes] = applyDA(pDoentes, numberAugmentedSickImages, filesDoentes);
        allDoentes = [doentesData; augmentedDoentes];
    else
       allSaudaveis = saudaveisData;
       allDoentes = doentesData;
    end
    
    allImages = [allSaudaveis; allDoentes];
    
    maxBase = max(allImages(:))
    minBase = min(allImages(:))
    
    pSaudaveisRatio = preprocessAndSaveImages(pSaudaveis, minBase, maxBase, filesSaudaveis, toFolderName, folderClassSaudaveis, allSaudaveisFileNames);
    pDoentesRatio = preprocessAndSaveImages(pDoentes, minBase, maxBase, filesDoentes, toFolderName, folderClassDoentes, allDoentesFileNames);
    
    if hasDataAugment
        pSaudaveisAugmentedRatio = preprocessAndSaveAugmentedImages(pAugmentedSaudaveis, minBase, maxBase, filesSaudaveis, toFolderName, folderClassSaudaveis, allSaudaveisFileNames, numberAugmentedHealthImages);
        pDoentesAugmentedRatio = preprocessAndSaveAugmentedImages(pAugmentedDoentes, minBase, maxBase, filesDoentes, toFolderName, folderClassDoentes, allDoentesFileNames, numberAugmentedSickImages);
    end
    

end

function [images, pData, allFileNames] = readImages(files, folderName)

    sizeSet = size(files,1);
    pData = cell(sizeSet,1);
    images = zeros(sizeSet, 480, 640);
    allFileNames = cell(sizeSet,1);

    for i = 1:sizeSet
        
        fileName=files(i).name;
        fullPath = strcat(folderName, '/', fileName);
        img = load(fullPath);

        splittedFileName = split(fileName, '.txt');
        fileName = splittedFileName{1};
        
        allFileNames{i} = fileName;

        pData{i} = img;
        
        images(i, :, :) = img;
    end

end


function [alteredData,pAlteredData] = applyDA(data, numberAugmentedImages, files)
    sizeSet = size(files,1);
    alteredData = zeros(sizeSet*numberAugmentedImages, 480, 640);
    pAlteredData = cell(sizeSet*numberAugmentedImages,1);
    
    for i = 1:sizeSet
        img = data{i};
        start = ((i-1)*numberAugmentedImages + 1);
        endV = i*numberAugmentedImages;
%         disp([start, endV])
        [alteredData(start : endV, :, :, :), pAlteredData( start : endV)] = dataAugmentNoSaving(img, numberAugmentedImages);
        close all
    end
end

function [imagedPreProcessed, numpyImg] = preprocessImage(img, minValue, maxValue, fileName)

    imgRatio = (255*(img - minValue))/(maxValue-minValue);
    imgRatio = round(imgRatio, 0);
    imgRatio = imgRatio/255;

    imagedPreProcessed(:,:,1) = imgRatio;
    imagedPreProcessed(:,:,2) = imgRatio;
    imagedPreProcessed(:,:,3) = imgRatio;
    
    figure;
    subplot(1,2,1)
    imagesc(imgRatio);
    title(fileName)

    subplot(1,2,2)
    histogram(imgRatio);

    folderSaveData = strcat('imagesVerify/', fileName, '.png');
    saveas(gcf, folderSaveData)
    
    figure;
    
    subplot(1,2,1)
    imagesc(imgRatio);
    title(fileName)

    subplot(1,2,2)
    histogram(imgRatio);
    numpyImg = py.numpy.array(imgRatio);
end

function pData = preprocessAndSaveImages(data, minValue, maxValue, files, toFolderName, folderClass, fileNames)
    sizeSet = size(files,1);
    pData = cell(sizeSet,1);
    
    for i = 1:sizeSet
        img = data{i};
        fileName = fileNames{i};
        
        [imgRatio, numpyImg] = preprocessImage(img, minValue, maxValue, fileName);
        pData{i} = imgRatio;
       
        %Saving image 
        
        folderSaveData = strcat('../../',toFolderName,'/', folderClass, '/',fileName);
        py.numpy.save(folderSaveData, numpyImg);
        
        close all
    end
end



function pData = preprocessAndSaveAugmentedImages(data, minValue, maxValue, files, toFolderName, folderClass, fileNames, numberAugmentedImages)
    sizeSet = size(files,1);
    pData = cell(sizeSet,1);
    totalImages = sizeSet*numberAugmentedImages;
    indexFile = 1;
    
    for i = 1:totalImages
        img = data{i};
        
        valueMod = mod(i, numberAugmentedImages);
        fileName = fileNames{indexFile};
        fileName = strcat(fileName, '_alt_', num2str(valueMod));
        if valueMod == 0
            indexFile = indexFile+1;
        end
        
        [imgRatio, numpyImg] = preprocessImage(img, minValue, maxValue, fileName);
        pData{i} = imgRatio;
       
        %Saving image 
        
        folderSaveData = strcat('../../',toFolderName,'/', folderClass, '/',fileName);
        py.numpy.save(folderSaveData, numpyImg);
        
        close all
    end
end
