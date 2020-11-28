from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *
from ag_cnnInit import *
from ag_cnnFromAG import *

tp=4
tour=3
tr=80
tm=50
sequenceIndividual = [i for i in range(12)]
population = initializePopulation(tp)

populationFitness = [6,8,10,4]#,5,9]
selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)

newPopulationAfterMutation = applyMutation(newPopulation, tm, tp)

#cnn parts
isNumpy=True
trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, device, criterion = prepareCNN(isNumpy)
individuo = population[0]
model, optimizer, epocas = convertAgToCNN(individuo, device)

# individuo = population[0]
# convertAgToCNN(individuo, device)

# individuo = population[0]
# individuoSize = len(individuo)
# print('individuoSize', individuoSize)
# layersAsArray = []
# for i in range(2, individuoSize, 2):
#     print('i', i)
#     geneLayer = individuo[i]
#     #tem a camada dessa layer
#     if geneLayer[0] == 1:
#         geneDropout = individuo[i+1]
#         layersAsArray.extend(defineSingleLayer(nInputs, geneLayer, geneDropout))
#         nInputs = geneLayer[1]

# childPopulation=newPopulation#.copy()
# sequenceChild = [i for i in range(len(childPopulation))]
    
# numberChildrenToMutate = int(tm * tp /100)
# individualsToMutate = sample(sequenceChild, numberChildrenToMutate)
# print('Individuos para mutacao', individualsToMutate)

# mutatedIndividuals = [applyMutationInd(childPopulation[i], sequenceChild) for i in individualsToMutate]
