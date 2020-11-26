
function dataAugment2DImage(imgMinMax, nomePacientes, patientIndex, numberAltered, folderImageName, folderName, folderDirectory)
    for i = 1:numberAltered
        %getting mean
        I = imgMinMax;
        thresh = multithresh(I);
        seg_I = imquantize(I,thresh);
        %figure; imagesc(seg_I);

        meanValue1 = mean(imgMinMax(seg_I == 1));
        
        disp('na funcao')
        min(imgMinMax(:))
        max(imgMinMax(:))
        
        %Generating and saving one altered image
        imOriginal = imgMinMax;
        tform = randomAffine2d('Rotation',[-45 45], 'XReflection',true,'YReflection',true); 
        outputView = affineOutputView(size(imOriginal),tform);
        imAlterada = imwarp(imOriginal,tform,'OutputView',outputView,'FillValues',meanValue1);
    %     imAlteradaCor = jitterColorHSV(imAlterada,'Contrast',[1.2 1.4],'Saturation',[-0.4 -0.1],'Brightness',[-0.2 0.2]);
        imAlteradaCor=imAlterada;
        figure;
        imagesc(imAlteradaCor);
        min(imgMinMax(:))
        max(imgMinMax(:))
        
       
        folderSaudaveis = strcat(folderImageName, nomePacientes(patientIndex, :), '_alt',string(i),'.png');
        saveas(gcf, folderSaudaveis)

        numpyRGB = py.numpy.array(imAlteradaCor);
%         folderSaudaveis = strcat('../../',folderDirectory,'/',folderName,'/', nomePacientes(patientIndex, :), '_alt_', string(i));
%         py.numpy.save(folderSaudaveis, numpyRGB);
    end
end