# Infrared termography and convolutional neural networks

## Estrutura do código
O repositório contém códigos correspondentes a cinco fluxos:

- Experimentos manuais (Experimentos manuais são os experimentos onde a parte densamente conectada da rede é definida manualmente e empiricamente)
- AG 
- PSO
- Surrogate com AG
- Surrogate com PSO

## Quais dados utilizar

Para definir qual a base de dados a ser utilizada, o caminho das pastas está definido diretamente nos arquivos:

- prepareDataDictionary.py -> para bases de dados que são txt
- readMatlabNumpyData.py -> para bases de dados que foram tratadas e estão em numpy arrays.

A definição de qual dos dois nomes de pastas será utilizado é definida pelo parâmetro isNumpy que é um argumento em todas as funções principais.
Atenção ao caminho do colab, que espera que haja uma pasta chamada `MestradoCodes` dentro da pasta do `Meu Drive`.

## Pre processamento 

Mais detalhes sobre o pre processamento podem ser encontrados [aqui](Pre%20processamento.md)
## Rodar experimentos manuais no Google Colab:

```python
from mainDensenet import *

resultsPlotName = 'plotsName'
experimentType = 1
typeLR = 2

# mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy=True, nEpochs=30, maxEpochs=None)
model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainDensenet(resultsPlotName, experimentType, False, typeLR, False)
```

### Exemplos

Para mais detalhes do AG e do PSO consulte [aqui](AG%20e%20PSO.md)

A pasta [Colab file example](Colab%20file%20example/Exemplos.md) contém um exemplo de arquivo .ipynb para cada um dos fluxos.
