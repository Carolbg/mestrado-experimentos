function [augmentedImages, pAugmentedImages] = dataAugmentNoSaving(img, numberAltered)
    
    augmentedImages = ones(numberAltered, 480, 640);
    pAugmentedImages = cell(numberAltered,1);
    
    for i = 1:numberAltered
        I = img;
        thresh = multithresh(I, 3);
        seg_I = imquantize(I,thresh);
        figure; imagesc(seg_I);

        meanValue = mean(img(seg_I == 1));

        %Generating and saving one altered image
        imOriginal = img;
        tform = randomAffine2d('Rotation',[-45 45], 'XReflection',true,'YReflection',true); 
        outputView = affineOutputView(size(imOriginal),tform);
        imgAlterada = imwarp(imOriginal,tform,'OutputView',outputView,'FillValues',[meanValue]);
        
        figure;
        imagesc(imgAlterada);
        
        augmentedImages(i, :, :) = imgAlterada;
        pAugmentedImages{i} = imgAlterada;
    end
end
