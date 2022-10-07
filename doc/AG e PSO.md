# Rodar otimização AG e PSO:

Nesta sessão serão exemplificados como rodar o AG e o PSO, com os respectivos parâmetros de cada algoritmo.

Lembre-se de conferir se o nome da pasta é dos dados que você gostaria de executar. 

## Rodar AG:

```python
from ag_main import *

#read the data only once to use in all experiments
readData(False, 30)

# run the ag with this configuration
# main(tp=10, tour=2, tr=80, numberIterations=10, tm=40, isNumpy=True, cnnType=1, useSurrogate=False)
# cnnType: 1 = resnet, 2 = VGG, 3 = Densenet
population, populationFitness, model = main(10, 2, 80, 10, 40, False, 3, False)

```

O último parâmetro define o uso ou não do surrogate. No exemplo, o AG não tem o surrogate habilitado, porém para habilitar basta trocar o último parâmetro.

## Rodar PSO:

```python
from pso import *

# read the data only once to use in all experiments
# readData(isNumpy=False, nEpochs=30)
readData(False, 30)

# PSO(iterations=10, populationSize=10, Cg=0.5, cnnType=1, useSurrogate=False)
# cnnType: 1 = resnet, 2 = VGG, 3 = Densenet
swarm = PSO(10, 10, 0.6, 1, False)
```

Equivalente ao AG, o último parâmetro do PSO define o uso ou não do surrogate.