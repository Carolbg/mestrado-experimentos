from ag_fitness import saveGlobalVariables
import numpy as np
import timeit
from randomForest import testModel
import multiprocessing as mp
from ag_verifyLayers import verifyNetworkLayers
from surrogate_encoding import *

def calcSurrogateFitness(randomTreeModel, generation, population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType):
    print('\n\n@@@@ Calculando fitness com surrogate')
    tp = len(population)
    # print('calcFitness', trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
    
    # arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayCacheConfigClass, arrayMaxEpochsStop, arrayNEpochs, arrayCnnType = saveGlobalVariables(generation, 
    #     trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, tp, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
    
    startAll = timeit.default_timer()
    iterations = [i for i in range(tp)]
    
    fitnessArray = []
    # try:
    #     mp.set_start_method('spawn')
    # except:
    #     print('error')
    
    print('after error', max_epochs_stop, n_epochs)

    fitnessArray = []

    for individual in population:
        cacheValue = cacheConfigClass.verifyEntry(individual)
        if cacheValue != None:
            # print('\nachei cache', cacheValue, ' individual = ', individual, '\n fitness = ', cacheValue)
            fitnessArray.append(cacheValue)
            # print('cacheValue', cacheValue)
        else:
            encodedIndividual = encodeAGIndividual(individual)
            npData = np.array(encodedIndividual)
            flattenIndividual = npData.flatten()
            # print('flattenIndividual', flattenIndividual)
            fitnessIndividual = testModel(randomTreeModel, [flattenIndividual])
            # print('resultado rf', fitnessIndividual)
            fitnessArray.append(fitnessIndividual[0])
    
    # print('fitnessArray', fitnessArray)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for result in zip(iterations, executor.map(calcFitnessIndividuo, population, iterations, arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayCacheConfigClass,  arrayMaxEpochsStop, arrayNEpochs, arrayCnnType)):
    #         iteration, (fitness, _) = result
    #         print('result', iteration, fitness)
    #         fitnessArray.append(fitness)


    # fitnessArray = [calcFitnessIndividuo(population[i], i, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion) for i in range(tp)]
    return np.array(fitnessArray)

def calcRandomForestFitnessIndividuo(randomTreeModel, individuo, i, generation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType, fileName=None):
    print('calcRandomForestFitnessIndividuo = ', i, individuo, '\n')
    # print('\n pt1 ', i, trainLoader, testLoader, validationLoader)
    # print('\n pt2', cat_df, batch_size, device, criterion)
    print('cacheConfigClass', cacheConfigClass)
    print('device', device, 'maxepoch', max_epochs_stop)
    cacheValue = cacheConfigClass.verifyEntry(individuo)
    if cacheValue != None:
        print('\nachei cache', cacheValue, ' individuo = ', i, individuo, '\n fitness = ', cacheValue)
        return cacheValue

    fitnessIndividual = testModel(randomTreeModel, individuo)
    
    isReducingLayerSize = verifyNetworkLayers(individuo) 

    if isReducingLayerSize == False:
        fitnessIndividual = fitnessIndividual*0.7

    return fitnessIndividual

def calcSurrogatePSOFitness(randomTreeModel, swarm, cacheConfigClass, max_epochs_stop, n_epochs, halfPopulation):
    print('\n\n@@@@ Calculando fitness pso com surrogate')

    # print('calcFitness', trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
    
    # arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayCacheConfigClass, arrayMaxEpochsStop, arrayNEpochs, arrayCnnType = saveGlobalVariables(generation, 
    #     trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, tp, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
    
    startAll = timeit.default_timer()
    # iterations = [i for i in range(tp)]
    
    fitnessArray = []
    # try:
    #     mp.set_start_method('spawn')
    # except:
    #     print('error')
    
    print('after error', max_epochs_stop, n_epochs)

    fitnessArray = []
    
    print('halfPopulation', halfPopulation)
    
    for i in range(halfPopulation, len(swarm)):
        print('i = ', i)
        cacheValue = cacheConfigClass.verifyEntry(swarm[i]['position'])
        if cacheValue != None:
            # print('\nachei cache', cacheValue, ' individual = ', individual, '\n fitness = ', cacheValue)
            fitnessArray.append(cacheValue)
            # print('cacheValue', cacheValue)
        else:
            encodedIndividual = encodeParticle(swarm[i]['bestPosition'])
            npData = np.array(encodedIndividual)
            flattenIndividual = npData.flatten()
            # print('flattenIndividual', flattenIndividual)
            fitnessIndividual = testModel(randomTreeModel, [flattenIndividual])
            # print('resultado rf', fitnessIndividual)
            # fitnessArray.append(fitnessIndividual[0])
            swarm[i]['positionFitness'] = fitnessIndividual[0]
    
    # print('fitnessArray', fitnessArray)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for result in zip(iterations, executor.map(calcFitnessIndividuo, population, iterations, arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayCacheConfigClass,  arrayMaxEpochsStop, arrayNEpochs, arrayCnnType)):
    #         iteration, (fitness, _) = result
    #         print('result', iteration, fitness)
    #         fitnessArray.append(fitness)


    # fitnessArray = [calcFitnessIndividuo(population[i], i, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion) for i in range(tp)]
    return np.array(fitnessArray)

