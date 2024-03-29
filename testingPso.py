from pso_initialize import *
from pso_fitness import *
from utils_readAllData import *
from pso import *
from simpleIteration import *
from psoCacheClass import PSOCacheClass

populationSize=5
Cg=0.7

from pso import *
swarm = PSO(10, 20, 0.7, False,1, 1)

isNumpy=False
cnnType=1
nEpochs=2
readData(isNumpy, nEpochs)
trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()
swarm = initializeSwarm(populationSize)
cacheConfigClass = PSOCacheClass()
calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType, cacheConfigClass)
# print(swarm)

for particle in swarm:
    # print('swarm no init', swarm)
    updateBestSolutionParticle(particle)

swarm = initializeSwarm(populationSize)
diffPBest = calcDiffTwoParticles(swarm[1]['position'], swarm[0]['position'])
diffGBest = calcDiffTwoParticles(swarm[2]['position'], swarm[0]['position'])
newVelocity = calcVelocity(Cg, diffPBest, diffGBest, swarm[1]['position'], swarm[2]['position'])
a = updateParticlePosition(swarm[0]['position'], newVelocity)

########################
from pso_initialize import *
from pso_fitness import *
from utils_readAllData import *
from pso import *
from simpleIteration import *
populationSize=5
Cg=0.7
isNumpy=False
cnnType=1
nEpochs=2
readData(isNumpy, nEpochs)
trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()
swarm = initializeSwarm(populationSize)
cacheConfigClass = PSOCacheClass()
calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType, cacheConfigClass)
# print(swarm)

for particle in swarm:
    # print('swarm no init', swarm)
    updateBestSolutionParticle(particle)

swarm = singleIteration(swarm, populationSize, Cg)


#calc das redes novas geradas com o update das posicoes
cacheConfigClass = PSOCacheClass()
calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType, cacheConfigClass)
for particle in swarm:
    updateBestSolutionParticle(particle)
    
swarm = bestNeighbourPosition(swarm, populationSize)
print('\n')


######
from pso_initialize import *
from psoCacheClass import PSOCacheClass

populationSize=5
swarm = initializeSwarm(populationSize)
cacheConfigClass = PSOCacheClass()

particle = swarm[0]
particlePosition = particle['position']
cacheValue = cacheConfigClass.verifyEntry(particlePosition)
cacheConfigClass.savePopulationToCache(swarm)
