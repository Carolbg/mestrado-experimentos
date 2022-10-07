# Pre processamento

O preprocessamento dos dados pode ser feito tanto no próprio código python ou no matlab.
No geral, os aumentos de dados utilizados nos resultados publicados foram gerados utilizando os códigos do matlab. Além disso, o preprocessamento do Jet colormap necessariamente é realizado utilizando o Matlab.

O parâmetro dos algoritmos isNumpy se refere a dados que foram previamente tratados no matlab e foram salvos como numpy array para serem lidos no fluxo das CNNs.

## Matlab Codes:
Alguns códigos matlab são o primeiro passo e devem ser executados antes de executar o código python devido ao fato de que eles geram os dados que as CNNs no python usam.

### Funções
- preProcessImagesJetColormap -> Realiza o pre processamento utilizando o jet colormap  no matlab e salva os numpy array. 
    - Parametros: 
        - hasDataAugment -> true ou false para indicar se chama ou nao aumento de dados
        - originFolderName -> nome da pasta onde estao os dados originais
        - toFolderName -> nome da pasta onde vai salvar os arquivos
            - A pasta precisa estar criada e com duas subpastas dentro:
                0Saudavel
                1Doente
        - numberAugmentedHealthImages -> numero de imagens aumentadas do conjunto saudaveis
        - numberAugmentedSickImages -> numero de imagens aumentadas do conjunto doentes
        - Ex: [pSaudaveisRGB, pDoentesRGB] = preProcessImagesJetColormap(true, 'Imagens_TXT_Estaticas_Balanceadas_asCabıoglu', 'teste', 1, 7);

    - Lê todos os arquivos da pasta
    - Pode ou não chamar a geração de aumento de dados (via parametro)
    - Não usa o nem min max e nem ratio, já que o range dos valores com a conversão do jet resulta em valores entre 0 e 1
    - A conversao para (C, H, W) no formato do tensor eh feita internamente no python
    - Versao melhorada de colormap_readingAllFiles

- preProcessImagesRatio -> Realiza o pre processamento utilizando o ratio no matlab e salva os numpy array. 
    - Parametros: 
        - hasDataAugment -> true ou false para indicar se chama ou nao aumento de dados
        - originFolderName -> nome da pasta onde estao os dados originais
        - toFolderName -> nome da pasta onde vai salvar os arquivos
        - numberAugmentedHealthImages -> numero de imagens aumentadas do conjunto saudaveis
        - numberAugmentedSickImages -> numero de imagens aumentadas do conjunto doentes
        - Ex: [pSaudaveisRatio, pSaudaveisAugmentedRatio, pDoentesRatio, pDoentesAugmentedRatio] = preProcessImagesRatio(true, 'Imagens_TXT_Estaticas_Balanceadas_asCabıoglu', 'teste', 1, 2);

    - Lê todos os arquivos da pasta
    - Pode ou não chamar a geração de aumento de dados (via parametro)
    - A matriz final eh uma matriz 3D com valores entre 0 e 1 e (480, 640, 3)
    - A conversao para (C, H, W) no formato do tensor eh feita internamente no python
    
### Scripts

- minMax_preprocessing -> Realiza o pre processamento utilizando o ratio (similar ao min max) no matlab e salva os numpy array. 
    - Pode ou não chamar a geração de aumento de dados
- colormap_preprocessing.m -> Realiza o pre processamento utilizando o jet colormap  no matlab e salva os numpy array. 
    - Pode ou não chamar a geração de aumento de dados
    - Não usa o min max nem ratio já que o range dos valores com a conversão do jet resulta em valores entre 0 e 1
- colormap_readingAllFiles -> Realiza o pre processamento utilizando o jet colormap  no matlab e salva os numpy array. 
    - A diferença desse para o de cima é que ele lê todos os arquivos da pasta enquanto o de cima lê os arquivos com nome especificados no array.
    - Pode ou não chamar a geração de aumento de dados
    - Não usa o min max nem ratio já que o range dos valores com a conversão do jet resulta em valores entre 0 e 1
- ratio_preprocessing
