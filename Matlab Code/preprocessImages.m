%% SAUDAVEL Paciente 221
imagesSaudaveis = [
    'T0221.1.1.S.2013-11-18.00';
    'T0221.1.2.S.2013-11-18.00';
    'T0221.1.3.S.2013-11-18.00';
    'T0221.1.4.S.2013-11-18.00';
];

saudaveis = './Imagens_TXT_Estaticas_Balanceadas_cortadas/0Saudavel/';
sizeSaudaveis = size(imagesSaudaveis,1);
pSaudaveis = cell(sizeSaudaveis,1);
for i = 1:sizeSaudaveis
    fullPath = strcat(saudaveis, imagesSaudaveis(i, :), '.txt');
    img = load(fullPath); 
    pSaudaveis{i} = img;
end

pSaudaveis{1} = pSaudaveis{1}(1:400, :);
pSaudaveis{2} = pSaudaveis{2}(1:387, :);
pSaudaveis{3} = pSaudaveis{3}(1:397, :);
pSaudaveis{4} = pSaudaveis{4}(1:390, :);

fig = figure;

for i = 1:sizeSaudaveis
    fileName = strcat(saudaveis, imagesSaudaveis(i, :),  '_cropped.txt');
    %imwrite(pSaudaveis{i}, strcat(saudaveis, imagesSaudaveis(i, :),'.png'));
    fileID = fopen(fileName,'w');
    subplot(3,2,i)
    imagesc(pSaudaveis{i})
    for j = 1:size(pSaudaveis{i}, 1)
        for k = 1:size(pSaudaveis{i}, 2)
            if k ~= 1
                fprintf(fileID, ' %.2f', pSaudaveis{i}(j,k));
            else
                fprintf(fileID, '%.2f', pSaudaveis{i}(j,k));
            end
        end
        if j < size(pSaudaveis{i}, 1)
            fprintf(fileID, '\n');
        end
    end
         
end

saveas(fig,'T0221_cropped.png')

%% SAUDAVEL Paciente 190
imagesSaudaveis = [
    'T0190.1.1.S.2013-09-30.00';
    'T0190.1.2.S.2013-09-30.00';
    'T0190.1.3.S.2013-09-30.00';
    'T0190.1.4.S.2013-09-30.00';
    'T0190.1.5.S.2013-09-30.00';
];

saudaveis = './Imagens_TXT_Estaticas_Balanceadas_cortadas/0Saudavel/';
sizeSaudaveis = size(imagesSaudaveis,1);
pSaudaveis = cell(sizeSaudaveis,1);
for i = 1:sizeSaudaveis
    fullPath = strcat(saudaveis, imagesSaudaveis(i, :), '.txt');
    img = load(fullPath); 
    pSaudaveis{i} = img;
end

pSaudaveis{1} = pSaudaveis{1}(1:433, :);
pSaudaveis{2} = pSaudaveis{2}(1:423, :);
pSaudaveis{3} = pSaudaveis{3};
pSaudaveis{4} = pSaudaveis{4}(1:411, :);
pSaudaveis{5} = pSaudaveis{5};

fig = figure;

for i = 1:sizeSaudaveis
    fileName = strcat(saudaveis, imagesSaudaveis(i, :),  '_cropped.txt');
    fileID = fopen(fileName,'w');
    subplot(3,2,i)
    imagesc(pSaudaveis{i})
    for j = 1:size(pSaudaveis{i}, 1)
        for k = 1:size(pSaudaveis{i}, 2)
            if k ~= 1
                fprintf(fileID, ' %.2f', pSaudaveis{i}(j,k));
            else
                fprintf(fileID, '%.2f', pSaudaveis{i}(j,k));
            end
        end
        if j < size(pSaudaveis{i}, 1)
            fprintf(fileID, '\n');
        end
    end
         
end

saveas(fig,'T0190_cropped.png')

%% SAUDAVEL Paciente 191
imagesSaudaveis = [
    'T0191.1.1.S.2013-09-06.00';
    'T0191.1.2.S.2013-09-06.00';
    'T0191.1.3.S.2013-09-06.00';
    'T0191.1.4.S.2013-09-06.00';
    'T0191.1.5.S.2013-09-06.00';
];

saudaveis = './Imagens_TXT_Estaticas_Balanceadas_cortadas/0Saudavel/';
sizeSaudaveis = size(imagesSaudaveis,1);
pSaudaveis = cell(sizeSaudaveis,1);
for i = 1:sizeSaudaveis
    fullPath = strcat(saudaveis, imagesSaudaveis(i, :), '.txt');
    img = load(fullPath); 
    pSaudaveis{i} = img;
end

pSaudaveis{1} = pSaudaveis{1}(1:411,:);
pSaudaveis{2} = pSaudaveis{2}(1:425,:);
pSaudaveis{3} = pSaudaveis{3}(1:420,:);
pSaudaveis{4} = pSaudaveis{4}(1:425,:);
pSaudaveis{5} = pSaudaveis{5}(1:445,:);

fig = figure;

for i = 1:sizeSaudaveis
    fileName = strcat(saudaveis, imagesSaudaveis(i, :),  '_cropped.txt');
    fileID = fopen(fileName,'w');
    subplot(3,2,i)
    imagesc(pSaudaveis{i})
    for j = 1:size(pSaudaveis{i}, 1)
        for k = 1:size(pSaudaveis{i}, 2)
            if k ~= 1
                fprintf(fileID, ' %.2f', pSaudaveis{i}(j,k));
            else
                fprintf(fileID, '%.2f', pSaudaveis{i}(j,k));
            end
        end
        if j < size(pSaudaveis{i}, 1)
            fprintf(fileID, '\n');
        end
    end
         
end

saveas(fig,'T0191_cropped.png')

%% DOENTE Paciente 138
imagesDoentes = [
    'T0138.2.1.S.2013-09-06.00';
    'T0138.2.2.S.2013-09-06.00';
    'T0138.2.3.S.2013-09-06.00';
    'T0138.2.4.S.2013-09-06.00';
    'T0138.2.5.S.2013-09-06.00';
];

doentes = './Imagens_TXT_Estaticas_Balanceadas_cortadas/1Doente/';
sizeDoentes = size(imagesDoentes,1);
pDoentes = cell(sizeDoentes,1);
for i = 1:sizeDoentes
    fullPath = strcat(doentes, imagesDoentes(i, :), '.txt');
    img = load(fullPath); 
    pDoentes{i} = img;
end

pDoentes{1} = pDoentes{1}(1:359,:);
pDoentes{2} = pDoentes{2}(1:359,:);
pDoentes{3} = pDoentes{3}(1:357,:);
pDoentes{4} = pDoentes{4}(1:356,:);
pDoentes{5} = pDoentes{5}(1:345,:);

fig = figure;

for i = 1:sizeDoentes
    fileName = strcat(doentes, imagesDoentes(i, :),  '_cropped.txt');
    fileID = fopen(fileName,'w');
    subplot(3,2,i)
    imagesc(pDoentes{i})
    for j = 1:size(pDoentes{i}, 1)
        for k = 1:size(pDoentes{i}, 2)
             if k ~= 1
                fprintf(fileID, ' %.2f', pDoentes{i}(j,k));
            else
                fprintf(fileID, '%.2f', pDoentes{i}(j,k));
            end
        end
        if j < size(pDoentes{i}, 1)
            fprintf(fileID, '\n');
        end
    end
         
end

saveas(fig,'T0138_cropped.png')

%% DOENTE Paciente 179
imagesDoentes = [
    'T0179.1.1.S.2013-08-16.00';
    'T0179.1.2.S.2013-08-16.00';
    'T0179.1.3.S.2013-08-16.00';
    'T0179.1.4.S.2013-08-16.00';
    'T0179.1.5.S.2013-08-16.00';
];

doentes = './Imagens_TXT_Estaticas_Balanceadas_cortadas/1Doente/';
sizeDoentes = size(imagesDoentes,1);
pDoentes = cell(sizeDoentes,1);
for i = 1:sizeDoentes
    fullPath = strcat(doentes, imagesDoentes(i, :), '.txt');
    img = load(fullPath); 
    pDoentes{i} = img;
end

pDoentes{1} = pDoentes{1}(1:430,:);
pDoentes{2} = pDoentes{2}(1:423,:);
pDoentes{3} = pDoentes{3}(1:435,:);
pDoentes{4} = pDoentes{4}(1:404,:);
pDoentes{5} = pDoentes{5}(1:422,:);
fig = figure;

for i = 1:sizeDoentes
    fileName = strcat(doentes, imagesDoentes(i, :),  '_cropped.txt');
    fileID = fopen(fileName,'w');
    subplot(3,2,i)
    imagesc(pDoentes{i})
    for j = 1:size(pDoentes{i}, 1)
        for k = 1:size(pDoentes{i}, 2)
            if k ~= 1
                fprintf(fileID, ' %.2f', pDoentes{i}(j,k));
            else
                fprintf(fileID, '%.2f', pDoentes{i}(j,k));
            end
        end
        if j < size(pDoentes{i}, 1)
            fprintf(fileID, '\n');
        end
    end
         
end
saveas(fig,'T0179_cropped.png')

%% DOENTE Paciente 181
imagesDoentes = [
    'T0181.1.1.S.2013-08-16.00';
    'T0181.1.2.S.2013-08-16.00';
    'T0181.1.3.S.2013-08-16.00';
    'T0181.1.4.S.2013-08-16.00';
    'T0181.1.5.S.2013-08-16.00';
];

doentes = './Imagens_TXT_Estaticas_Balanceadas_cortadas/1Doente/';
sizeDoentes = size(imagesDoentes,1);
pDoentes = cell(sizeDoentes,1);
for i = 1:sizeDoentes
    fullPath = strcat(doentes, imagesDoentes(i, :), '.txt');
    img = load(fullPath); 
    pDoentes{i} = img;
end

pDoentes{1} = pDoentes{1}(1:455,:);
pDoentes{2} = pDoentes{2}(1:413,:);
pDoentes{3} = pDoentes{3}(1:418,:);
pDoentes{4} = pDoentes{4}(1:422,:);
pDoentes{5} = pDoentes{5}(1:410,:);
fig = figure;

for i = 1:sizeDoentes
    fileName = strcat(doentes, imagesDoentes(i, :),  '_cropped.txt');
    fileID = fopen(fileName,'w');
    subplot(3,2,i)
    imagesc(pDoentes{i})
    for j = 1:size(pDoentes{i}, 1)
        for k = 1:size(pDoentes{i}, 2)
            if k ~= 1
                fprintf(fileID, ' %.2f', pDoentes{i}(j,k));
            else
                fprintf(fileID, '%.2f', pDoentes{i}(j,k));
            end
        end
        if j < size(pDoentes{i}, 1)
            fprintf(fileID, '\n');
        end
    end
         
end

saveas(fig,'T0181_cropped.png')