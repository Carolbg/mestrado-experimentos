function [pSaudaveisRGB, pDoentesRGB] = preProcessImagesJetColormap(hasDataAugment, originFolderName, toFolderName, numberAugmentedHealthImages, numberAugmentedSickImages)

    %Leitura imagens saudaveis

    folderClass = '0Saudavel';
    nomeSaudaveis = strcat('../../', originFolderName, '/', folderClass);
    cd(nomeSaudaveis);
    filesSaudaveis=dir('*.txt');
    cd('../../Experimentos github/Matlab Code')

    pSaudaveisRGB = preprocessImageSet(hasDataAugment, numberAugmentedHealthImages, filesSaudaveis, nomeSaudaveis, toFolderName, folderClass);

    %Leitura imagens doentes

    folderClass = '1Doente';
    nomeDoentes = strcat('../../',originFolderName,'/', folderClass);
    cd(nomeDoentes);
    filesDoentes=dir('*.txt');
    cd('../../Experimentos github/Matlab Code')

   pDoentesRGB = preprocessImageSet(hasDataAugment, numberAugmentedSickImages, filesDoentes, nomeDoentes, toFolderName, folderClass);

end

function pDataRGB = preprocessImageSet(hasDataAugment, numberAugmentedImages, files, folderName, toFolderName, folderClass)
    sizeSet = size(files,1);
    pData = cell(sizeSet,1);
    pDataRGB = cell(sizeSet,1);
    minValue = ones(sizeSet);
    maxValue = ones(sizeSet);

    for i = 1:sizeSet
        
        fileName=files(i).name;
        fullPath = strcat(folderName, '/', fileName);
        img = load(fullPath);

        splittedFileName = split(fileName, '.txt');
        fileName = splittedFileName{1};

        pData{i} = img;

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

        minValue(i) = min(RGB(:));
        maxValue(i) = max(RGB(:));
        pDataRGB{i} = RGB;

        figure;
        subplot(1,2,1)
        imagesc(RGB);
        title(fileName)

        subplot(1,2,2)
        histogram(RGB);

        folderSaveData = strcat('imagesVerify/', fileName, '.png');
        saveas(gcf, folderSaveData)

        %Saving original image
        numpyRGB = py.numpy.array(RGB);
        folderSaveData = strcat('../../',toFolderName,'/', folderClass, '/',fileName);
        py.numpy.save(folderSaveData, numpyRGB);
        %
        if hasDataAugment
            dataAugment(img, RGB, fileName, i, numberAugmentedImages, 'imagesVerify/', folderClass, toFolderName)
        end
        
        close all
    end
end
