
%% Images definitions
nomeSaudaveis=[
    'T0177.1.1.S.2013-03-20.00',
    'T0188.1.1.S.2013-08-12.00',
    'T0189.1.1.S.2013-09-30.00',
    'T0190.1.1.S.2013-09-30.00',
    'T0191.1.1.S.2013-09-06.00',
    'T0193.1.1.S.2013-10-02.00',
    'T0194.1.1.S.2013-10-02.00',
    'T0195.1.1.S.2013-10-07.00',
    'T0196.1.1.S.2013-10-07.00',
    'T0199.1.1.S.2013-10-07.00',
    'T0200.1.1.S.2013-10-07.00',
    'T0201.1.1.S.2013-10-28.00',
    'T0208.1.1.S.2013-10-28.00',
    'T0211.1.1.S.2013-11-08.00',
    'T0212.1.1.S.2013-11-08.00',
    'T0216.1.1.S.2013-11-11.00',
    'T0217.1.1.S.2013-11-11.00',
    'T0218.1.1.S.2013-11-11.00',
    'T0219.1.1.S.2013-11-11.00',
    'T0220.1.1.S.2013-11-18.00',
    'T0221.1.1.S.2013-11-18.00',
    'T0222.1.1.S.2013-11-18.00',
    'T0224.1.1.S.2013-11-18.00',
    'T0225.1.1.S.2013-11-18.00',
    'T0226.1.1.S.2013-11-18.00',
    'T0233.1.1.S.2013-12-11.00',
    'T0234.1.1.S.2013-12-11.00',
    'T0236.1.1.S.2014-05-21.00',
    'T0237.1.1.S.2014-05-21.00',
    'T0238.1.1.S.2014-05-21.00',
    'T0239.1.1.S.2014-05-26.00',
    'T0243.1.1.S.2014-04-15.00',
    'T0244.1.1.S.2014-05-06.00',
    'T0259.1.1.S.2014-11-07.00',
    'T0261.1.1.S.2014-11-11.00',
    'T0272.1.1.S.2015-03-13.00',
    'T0275.1.1.S.2015-03-13.00',
    'T0276.1.1.S.2015-03-13.00',
];

nomeDoentes=[
    'T0138.2.1.S.2013-09-06.00',
    'T0179.1.1.S.2013-08-16.00',
    'T0180.1.1.S.2013-08-16.00',
    'T0181.1.1.S.2013-08-16.00',
    'T0192.1.1.S.2013-09-06.00',
    'T0198.2.1.S.2014-11-11.00',
    'T0202.1.1.S.2013-10-11.00',
    'T0203.1.1.S.2013-10-11.00',
    'T0204.1.1.S.2013-10-11.00',
    'T0209.1.1.S.2013-11-08.00',
    'T0210.1.1.S.2013-11-08.00',
    'T0213.1.1.S.2013-11-08.00',
    'T0240.1.1.S.2014-07-18.00',
    'T0241.1.1.S.2014-07-18.00',
    'T0245.1.1.S.2014-08-22.00',
    'T0246.1.1.S.2014-08-22.00',
    'T0255.1.1.S.2014-08-22.00',
    'T0256.1.1.S.2014-10-10.00',
    'T0257.1.1.S.2014-10-17.00',
    'T0258.1.1.S.2014-10-17.00',
    'T0260.1.1.S.2014-11-11.00',
    'T0263.1.1.S.2014-12-12.00',
    'T0264.1.1.S.2015-01-16.00',
    'T0266.1.1.S.2015-01-16.00',
    'T0267.1.1.S.2015-01-16.00',
    'T0268.1.1.S.2015-01-23.00',
    'T0269.1.1.S.2015-01-23.00',
    'T0270.1.1.S.2015-01-30.00',
    'T0271.1.1.S.2015-01-27.00',
    'T0273.1.1.S.2015-03-13.00',
    'T0277.1.1.S.2015-03-20.00',
    'T0278.1.1.S.2015-03-20.00',
    'T0281.1.1.S.2015-05-22.00',
    'T0282.1.1.S.2015-07-20.00',
    'T0283.1.1.S.2015-05-22.00',
    'T0285.1.1.S.2015-07-20.00',
    'T0286.1.1.S.2015-05-22.00',
    'T0287.1.1.S.2015-07-20.00',
];

%%
saudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/0Saudavel/';
sizeSaudaveis = size(nomeSaudaveis,1);

for i = 1:sizeSaudaveis
    fullPath = strcat(saudaveis, nomeSaudaveis(i, :), '.txt');
    img = load(fullPath);
    figure;
    imagesc(img)
    title(nomeSaudaveis(i, :))

end


%%
doentes = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/1Doente/';
sizeDoentes = size(nomeDoentes,1);

for i = 1:sizeDoentes
    fullPath = strcat(doentes, nomeDoentes(i, :), '.txt');
    img = load(fullPath);
    figure;
    imagesc(img)
    title(nomeDoentes(i, :))
end


%% Images definitions

removidosDoentes=[
    'T0192.1.1.S.2013-09-06.00',
    'T0256.1.1.S.2014-10-10.00',
    'T0258.1.1.S.2014-10-17.00',
    'T0263.1.1.S.2014-12-12.00',
];

doentes = '../../Imagens_TXT_Estaticas_Balanceadas_frontalImages/1Doente/';
sizeDoentes = size(removidosDoentes,1);

for i = 1:sizeDoentes
    fullPath = strcat(doentes, removidosDoentes(i, :), '.txt');
    img = load(fullPath);
    f = figure;
    imagesc(img)
    title(removidosDoentes(i, :))
    
%     folderDoentes = strcat('imagensDoentesRemovidas/', nomeDoentes(i, :), '.png');
%     saveas(gcf, folderDoentes)
    
end


%%
allSaudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/allSaudaveis/';
cd(allSaudaveis);
files=dir('*.txt');
cd('../../Experimentos github/Matlab Code')
for k=1:length(files)
    fileName=files(k).name;
   
    fullPath = strcat(allSaudaveis, fileName);
    img = load(fullPath);
    f = figure;
    imagesc(img)
    title(fileName)

end

%%
allSaudaveis = '../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/removidas saudaveis/';
cd(allSaudaveis);
files=dir('*.txt');
cd('../../Experimentos github/Matlab Code')
for k=1:length(files)
    fileName=files(k).name;
   
    fullPath = strcat(allSaudaveis, fileName);
    img = load(fullPath);
    f = figure;
    imagesc(img)
    title(fileName)
    
    folderDoentes = strcat('imagensSaudaveisRemovidas/', fileName, '.png');
    saveas(gcf, folderDoentes)

end