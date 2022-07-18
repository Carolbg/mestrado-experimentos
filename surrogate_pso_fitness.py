from ag_fitness import saveGlobalVariables
import numpy as np
import timeit
from randomForest import testModel
import multiprocessing as mp
from ag_verifyLayers import verifyNetworkLayers
from surrogate_encoding import *

def calcSurrogatePSOFitness(randomTreeModel, generation, population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType):
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
    
    return np.array(fitnessArray)
