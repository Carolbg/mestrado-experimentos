%%
A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0275.1.1.S.2015-03-13.00.txt') ; 
I = A;
figure;
imagesc(A);
title('original')
%%

J = medfilt2(I);
figure;
imagesc(J);
title('filtered')
% 
% 
% Nc = normalize(J);
% figure;
% imagesc(Nc);
% title('normalized')

%%

B = maxk(J(:),100);
meanTop10 = mean(B);

B = mink(J(:),100);
meanBottom10 = mean(B);

minMaxImg = (J - meanBottom10)/(meanTop10-meanBottom10);
figure;
imagesc(minMaxImg)
title('min max img')
