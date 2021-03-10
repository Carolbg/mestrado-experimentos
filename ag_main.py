from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *
from ag_cnnInit import *
from ag_fitness import *
from ag_reinsercao import *
from ag_readAllData import *
# import ag_cacheConfig

from cacheClass import CacheClass

def main(nEpochs=30, tp=10, tour=2, tr=80, numberIterations=10, tm=40, isNumpy=True, cnnType=1):
    startAll = timeit.default_timer()

    print('tp, tour, tr, numberIterations, tm, isNumpy', tp, tour, tr, numberIterations, tm, isNumpy)
    trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()
    # ag_cacheConfig.initCache()
    cacheConfigClass = CacheClass()
    sequenceIndividual = [i for i in range(11)]
    print('sequenceIndividual', sequenceIndividual)
    
    population = initializePopulation(tp)
    populationFitness = calcFitness(0, population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
    cacheConfigClass.savePopulationToCache(population, populationFitness)
    # print('\ncacheStore = ', ag_cacheConfig.cacheStore)

    for i in range(numberIterations):
        print('\n\n $$$$$$$$ Geração ', i)
        print('population = ', population)
        print('populationFitness = ', populationFitness)
        bestParent, bestParentFitness = findBestIndividuo(population, populationFitness)
        print('bestParent, bestParentFitness', bestParent, bestParentFitness)
        selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
        newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)

        newPopulation = applyMutation(newPopulation, tm, tp)
        # newPopulation = applyMutationPercentageForEachField(newPopulation, tm, tp)
        newPopulationFitness = calcFitness(i+1, newPopulation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
        
        # print('\n\n Saving new generated items of geração ', i)
        cacheConfigClass.savePopulationToCache(newPopulation, newPopulationFitness)

        population, populationFitness = selectNewGenerationDecrescente(bestParent, bestParentFitness, newPopulation, newPopulationFitness)
        
        # print('\ncacheStore = ', ag_cacheConfig.cacheStore)
    
    print('\nFinal population\n')
    print('population = ', population)
    print('populationFitness = ', populationFitness)
    bestParent, bestParentFitness = findBestIndividuo(population, populationFitness)
    print('bestParent, bestParentFitness', bestParent, bestParentFitness)

    bestParentModel = testingBestIndividuo(cnnType, bestParent, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs)

    endAll = timeit.default_timer()
    timeAll = endAll-startAll
    print('timeAll = ', timeAll)
    return population, populationFitness, bestParentModel

def testingBestIndividuo(cnnType, bestIndividuo, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, resultsPlotName='testDataResult'):
    cacheConfigClass = CacheClass()
    fitness, model = calcFitnessIndividuo(bestIndividuo, 'final', 'final', trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
    print('fitness', fitness)
    historyTest, cmTest = evaluate(model, testLoader, criterion, 2, resultsPlotName, device)
    print(cmTest)
    historyTest.to_csv('history_'+resultsPlotName+'.csv', index = False, header=True)
    
    return model
