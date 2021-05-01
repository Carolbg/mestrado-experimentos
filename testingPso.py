from pso_initialize import *
from pso_fitness import *
from utils_readAllData import *
from pso import *
populationSize=5
Cg=0.7


swarm = PSO(1, 5, 0.7, False,1, 2)

isNumpy=False
cnnType=1
nEpochs=2
readData(isNumpy, nEpochs)
trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()
swarm = initializeSwarm(populationSize)
calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType)
print(swarm)

for particle in swarm:
    print('swarm no init', swarm)
    updateBestSolutionParticle(particle)

swarm = initializeSwarm(populationSize)
diffPBest = calcDiffTwoParticles(swarm[1]['position'], swarm[0]['position'])
diffGBest = calcDiffTwoParticles(swarm[2]['position'], swarm[0]['position'])
newVelocity = calcVelocity(Cg, diffPBest, diffGBest, swarm[1]['position'], swarm[2]['position'])
a = updateParticlePosition(swarm[0]['position'], newVelocity)