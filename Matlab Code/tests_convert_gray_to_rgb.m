A = load('./Imagens_TXT_Estaticas/0Saudavel/T0001.1.1.S.2012-10-08.00.txt') ; 
imagesc(A)
rgbImage1 = cat(3, A, A, A);
rows = size(rgbImage1, 1);
colm = size(rgbImage1, 2);
depth = size(rgbImage1, 3);
rgbImage1Unica =reshape(rgbImage1,[1 rows*colm*depth]);
image(rgbImage1);

rgbImage2 = gray2rgb(A);
rows = size(rgbImage2, 1);
colm = size(rgbImage2, 2);
depth = size(rgbImage2, 3);
rgbImage2Unica =reshape(rgbImage2,[1 rows*colm*depth]);
image(rgbImage2)


% map = colormap(gray(256));
% rgbImage3 = ind2rgb(A, map);
% rows = size(rgbImage3, 1);
% colm = size(rgbImage3, 2);
% depth = size(rgbImage3, 3);
% rgbImage3Unica =reshape(rgbImage3,[1 rows*colm*depth]);
% image(rgbImage3)



B = load('./Imagens_TXT_Estaticas/1Doente/T0138.2.1.S.2013-09-06.00.txt'); 
imagesc(B)
brgbImage1 = cat(3, B, B, B);
rows = size(brgbImage1, 1);
colm = size(brgbImage1, 2);
depth = size(brgbImage1, 3);
brgbImage1Unica =reshape(brgbImage1,[1 rows*colm*depth]);
image(brgbImage1);

brgbImage2 = gray2rgb(B);
rows = size(brgbImage2, 1);
colm = size(brgbImage2, 2);
depth = size(brgbImage2, 3);
brgbImage2Unica =reshape(brgbImage2,[1 rows*colm*depth]);
image(brgbImage2)

map = [0 0 0.3
    0 0 0.4
    0 0 0.5
    0 0 0.6
    0 0 0.8
    0 0 1.0];

brgbImage3 = ind2rgb(B, map);
rows = size(brgbImage3, 1);
colm = size(brgbImage3, 2);
depth = size(brgbImage3, 3);
brgbImage3Unica =reshape(brgbImage3,[1 rows*colm*depth]);
image(brgbImage3)
