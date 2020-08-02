%% a
A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0275.1.1.S.2015-03-13.00.txt') ; 
I = A;
figure;
%% 
f = figure;
cmap = colormap(f,jet);
h = imagesc(A);
colorbar

%%
% 
% F = getimage;
% dadosFrame = F.cdata;

Cdata = h.CData;
cmap = colormap;

% make it into a index image.
cmin = min(Cdata(:));
cmax = max(Cdata(:));
m = length(cmap);

index = fix((Cdata-cmin)/(cmax-cmin)*m)+1; %A
% Then to RGB
RGB = ind2rgb(index,cmap);

figure;
subplot(1,2,1)
cmap = colormap(f,jet);
imagesc(RGB);
title('test')

subplot(1,2,2)
histogram(RGB);
saveas(gcf,'testing.png')
% numpyRGB = py.numpy.array(RGB);
% py.numpy.save('numpyTesting', numpyRGB);

%No python: x = np.load('numpyTesting.npy')