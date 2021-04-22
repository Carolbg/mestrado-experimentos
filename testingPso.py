from pso_initialize import *
from pso_fitness import *
from utils_readAllData import *
from pso import *

swarm = PSO(1, 5, 0.7, False,1, 2)

populationSize=2
Cg=0.7
isNumpy=False
cnnType=1
nEpochs=2
readData(isNumpy, nEpochs)
trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()
swarm = initializeSwarm(populationSize)
calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType)
print(swarm)
