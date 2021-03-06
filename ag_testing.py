from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *
from ag_cnnInit import *
from ag_cnnFromAG import *
from ag_fitness import *
from ag_reinsercao import *

tp=4
tour=3
tr=80
tm=50
sequenceIndividual = [i for i in range(11)]
population = initializePopulation(tp)

populationFitness = [6,8,10,4]#,5,9]
selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)

newPopulationAfterMutation = applyMutation(newPopulation, tm, tp)

#cnn parts
isNumpy=True
trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion = prepareCNN(isNumpy)
# individuo = population[0]
# model, optimizer, epocas = convertAgToCNN(individuo, device)
i=0
individuo = population[i]
model, optimizer, epocas = convertAgToCNN(individuo, device)
resultsPlotName = 'runAG_individuo_'+str(i)

#treinamento
model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
    optimizer, trainLoader, validationLoader, resultsPlotName, epocas, epocas, device)

#teste
historyTest, cmTest = evaluate(model, testLoader, criterion, 2, resultsPlotName, device)
testAcc = historyTest['test_acc'][0]

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

from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *
from ag_cnnInit import *
from ag_fitness import *
from ag_cacheConfig import *

# tp=3
# tour=2
# tr=100
# numberIterations=2
# tm=100
# isNumpy=True
tp=10
tour=3
tr=80
numberIterations=10
tm=20
isNumpy=True

trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion = prepareCNN(isNumpy)
initCache()
sequenceIndividual = [i for i in range(11)]

population = initializePopulation(tp)
populationFitness = calcFitness(population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)

for i in range(numberIterations):
    print('Geração ', i)
    selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
    newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)
    newPopulation = applyMutation(newPopulation, tm, tp)

    population = newPopulation
    populationFitness = calcFitness(newPopulation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)