# Infrared termography and convolutional neural networks

## Entry points
The code has some entry points:

- Entire flow in python
    - scripts 
    - colab
- Start flow in matlab for preprocessing and run python after
    - scripts 
- AGs
- PSOs


## Matlab Codes:
Some matlab code are the first step and should be run before running python code due to the fact that they generate the data the python might use.

### Funcoes
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

### Runing in Colab:

```python
from mainForColab import *
resultsPlotName = 'databaseMeanStdRatio_1camada_lr'
experimentType = 1
typeLR = 2
main(resultsPlotName, experimentType, typeLR)
```

### Running AG:

```python
from ag_main import *
from utils_readAllData import *

#read the data only once to use in all experiments
readData(False, 30)

#run the ag with this configuration
# main(tp=10, tour=2, tr=80, numberIterations=10, tm=40, isNumpy=True, cnnType=1)
population, populationFitness, model = main(10, 2, 80, 10, 40, True, 1)
```

### Running PSO:

```python
from pso import *

#read the data only once to use in all experiments
readData(False, 30)

#PSO(iterations=10, populationSize=10, Cg=0.5, isNumpy=False, cnnType=1, nEpochs=30)
swarm = PSO(10, 10, 0.5, False, 1, 30)
```
