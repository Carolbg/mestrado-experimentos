%% Generate images for article - health

A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.1.S.2013-03-20.00.txt') ; 
f = figure;
% cmap = colormap(f,jet);
h = imagesc(A);
colorbar
saveas(gcf,'health_frontal_174.png')




A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.2.S.2013-03-20.00.txt');
f = figure;
% cmap = colormap(f,jet);
h = imagesc(A);
colorbar
saveas(gcf,'health_lateral_174.png')
% 
% A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.3.S.2013-03-20.00.txt');
% A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.4.S.2013-03-20.00.txt');
% A = load('../../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/T0174.1.5.S.2013-03-20.00.txt');



%% Generate images for article - sick

A = load('../../Imagens_TXT_Estaticas_Balanceadas/1Doente/T0181.1.1.S.2013-08-16.00.txt') ; 
f = figure;
% cmap = colormap(f,jet);
h = imagesc(A);
colorbar
saveas(gcf,'sick_frontal_181.png')



A = load('../../Imagens_TXT_Estaticas_Balanceadas/1Doente/T0181.1.3.S.2013-08-16.00.txt');
f = figure;
% cmap = colormap(f,jet);
h = imagesc(A);
colorbar
saveas(gcf,'sick_lateral_181.png')
% 