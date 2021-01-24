%Verificar imagens validacao
allSaudaveis = '../../Removidos validacao/Removidos validacao saudaveis/';
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
    
    folderDoentes = strcat('imagensValidacaoSaudaveisRemovidas/', fileName, '.png');
    saveas(gcf, folderDoentes)

end

allSaudaveis = '../../Removidos validacao/Removidos validacao doentes/';
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
    
    folderDoentes = strcat('imagensValidacaoDoentesRemovidas/', fileName, '.png');
    saveas(gcf, folderDoentes)

end