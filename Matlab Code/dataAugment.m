function dataAugment(img, RGB, nomePacientes, patientIndex, numberAltered, folderImageName, folderName, folderDirectory)
    for i = 1:numberAltered
        %getting mean
        I = img;
        RGBParsed = RGB;
        thresh = multithresh(I);
        seg_I = imquantize(I,thresh);
        %figure; imagesc(seg_I);

        RGBParsed1 = RGBParsed(:,:,1);
        RGBParsed2 = RGBParsed(:,:,2);
        RGBParsed3 = RGBParsed(:,:,3);
        meanValue1 = mean(RGBParsed1(seg_I == 1));
        meanValue2 = mean(RGBParsed2(seg_I == 1));
        meanValue3 = mean(RGBParsed3(seg_I == 1));

        %Generating and saving one altered image
        imOriginal = RGB;
        tform = randomAffine2d('Rotation',[-45 45], 'XReflection',true,'YReflection',true); 
        outputView = affineOutputView(size(imOriginal),tform);
        imAlterada = imwarp(imOriginal,tform,'OutputView',outputView,'FillValues',[meanValue1 meanValue2 meanValue3]);
    %     imAlteradaCor = jitterColorHSV(imAlterada,'Contrast',[1.2 1.4],'Saturation',[-0.4 -0.1],'Brightness',[-0.2 0.2]);
        imAlteradaCor=imAlterada;
        figure;
        imagesc(imAlteradaCor);
        folderSaudaveis = strcat(folderImageName, nomePacientes(patientIndex, :), '_alt',string(i),'.png');
        saveas(gcf, folderSaudaveis)

        numpyRGB = py.numpy.array(imAlteradaCor);
        folderSaudaveis = strcat('../../',folderDirectory,'/',folderName,'/', nomePacientes(patientIndex, :), '_alt_', string(i));
        py.numpy.save(folderSaudaveis, numpyRGB);
    end
end