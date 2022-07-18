from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *
from utils_cnnInit import *
from ag_fitness import *
from ag_reinsercao import *
from utils_readAllData import *
# import ag_cacheConfig
from surrogate_main import *
from cacheClass import CacheClass
from surrogate_fitness import calcSurrogateFitness
import math

def main(tp=10, tour=2, tr=80, numberIterations=10, tm=40, isNumpy=True, cnnType=1, useSurrogate=False):
    startAll = timeit.default_timer()
    print('cnnType', cnnType)
    print('useSurrogate?', useSurrogate)
    #cnnType = 1 => resnet, cnnType = 2 => VGG, cnnType = 3 => Densenet
    print('tp, tour, tr, numberIterations, tm, isNumpy', tp, tour, tr, numberIterations, tm, isNumpy)
    print('fitness penalization = 0.6')
    trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()
    # ag_cacheConfig.initCache()
    cacheConfigClass = CacheClass()
    sequenceIndividual = [i for i in range(11)]
    print('sequenceIndividual', sequenceIndividual)
    
    halfPopulation = tp / 2
    halfPopulation = math.floor(halfPopulation)
    print('halfPopulation', halfPopulation)
    
    population = initializePopulation(tp)
    populationFitness = calcFitness(0, population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
    cacheConfigClass.savePopulationToCache(population, populationFitness)
    # print('\ncacheStore = ', ag_cacheConfig.cacheStore)
    allSurrogateTrainData = population
    allSurrogateTrainFitness = populationFitness
    randomForestModel = None
    
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
        print('i = ', i, 'i <= 1', i <= 1, 'useSurrogate', useSurrogate)
        if useSurrogate == False or i <= 1: 
            print('fitness apenas cnn')
            newPopulationFitness = calcFitness(i+1, newPopulation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
        else:
            #Real fitness
            newPopulationRealFitness = calcFitness(i+1, newPopulation[:halfPopulation], trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
            allSurrogateTrainData = np.concatenate((allSurrogateTrainData, newPopulation[:halfPopulation]))
            allSurrogateTrainFitness = np.concatenate((allSurrogateTrainFitness, newPopulationRealFitness))
            
            print('vou treinar a random forest with the new train data')
            randomForestModel = mainSurrogate(allSurrogateTrainData, allSurrogateTrainFitness)
            # print('randomForestModel treinado', randomForestModel)

            print('vou calcular fitness com surrogate')
            newPopulationSurrogateFitness = calcSurrogateFitness(randomForestModel, i+1, newPopulation[halfPopulation:], trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
            newPopulationFitness = np.concatenate((newPopulationRealFitness, newPopulationSurrogateFitness))

        # print('\n\n Saving new generated items of geração ', i)
        cacheConfigClass.savePopulationToCache(newPopulation, newPopulationFitness)

        population, populationFitness = selectNewGenerationDecrescente(bestParent, bestParentFitness, newPopulation, newPopulationFitness)
        
        if useSurrogate and i == 1:
            allSurrogateTrainData = np.concatenate((allSurrogateTrainData, population))
            allSurrogateTrainFitness = np.concatenate((allSurrogateTrainFitness, populationFitness))
            print('vou treinar a random forest')
            randomForestModel = mainSurrogate(allSurrogateTrainData, allSurrogateTrainFitness)
            # print('randomForestModel treinado', randomForestModel)

        # print('\ncacheStore = ', ag_cacheConfig.cacheStore)
    
    print('\nFinal population\n')
    print('population = ', population)
    print('populationFitness = ', populationFitness)
    bestParent, bestParentFitness = findBestIndividuo(population, populationFitness)
    print('bestParent, bestParentFitness', bestParent, bestParentFitness)

    n_epochs = 30
    max_epochs_stop = 10
    print('Testando com epocas ', n_epochs,' e maxEpocas', max_epochs_stop )
    bestParentModel = testingBestIndividuo(cnnType, bestParent, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, 'DataResult_n30_max10')

    n_epochs = 30
    max_epochs_stop = 30
    print('\n\n Sem early stopping - epocas ', n_epochs,' e maxEpocas', max_epochs_stop )
    bestParentModel = testingBestIndividuo(cnnType, bestParent, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, 'DataResult_n30_max30')

    n_epochs = 50
    max_epochs_stop = 10
    print('\n\Com early stopping - epocas ', n_epochs,' e maxEpocas', max_epochs_stop )
    bestParentModel = testingBestIndividuo(cnnType, bestParent, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, 'DataResult_n50_max10')

    n_epochs = 50
    max_epochs_stop = 50
    print('\n\n Sem early stopping com epocas ', n_epochs,' e maxEpocas', max_epochs_stop )
    bestParentModel = testingBestIndividuo(cnnType, bestParent, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, 'DataResult_n50_max50')

    endAll = timeit.default_timer()
    timeAll = endAll-startAll
    print('timeAll = ', timeAll)
    return population, populationFitness, bestParentModel

def testingBestIndividuo(cnnType, bestIndividuo, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, resultsPlotName):
    cacheConfigClass = CacheClass()
    trainName = 'finalTrain'+resultsPlotName
    fitness, model = calcFitnessIndividuo(bestIndividuo, 'final', 'final', trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType, trainName)
    print('fitness novo treinamento', fitness)
    testName = 'finalTest'+resultsPlotName
    historyTest, cmTest = evaluate(model, testLoader, criterion, 2, testName, device)
    print(cmTest)
    historyTest.insert(0, 'New Fitness', fitness, True)
    print(historyTest)
    
    historyTest.to_csv('history_'+resultsPlotName+'.csv', index = False, header=True)
    
    return model
