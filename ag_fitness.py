from ag_cnnFromAG import *
import ag_cacheConfig
from training import train
from testing import evaluate
# from cacheClass import cacheConfigClass
import copy
import timeit
import numpy as np
import concurrent.futures
from prepareDataDictionary import getCommonArgs
import multiprocessing as mp
from ag_verifyLayers import verifyNetworkLayers

def saveGlobalVariables(aGeneration, aTrainLoader, aTestLoader, aValidationLoader, aCat_df, aBatch_size, aDevice, aCriterion, tp, acacheConfigClass, amax_epochs_stop, an_epochs, acnnType):
    # global generation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion
    # generation = aGeneration
    # trainLoader = aTrainLoader
    # testLoader = aTestLoader
    # validationLoader = aValidationLoader
    # cat_df = aCat_df
    # batch_size = aBatch_size
    # device = aDevice
    # criterion = aCriterion
    arrayGeneration = [aGeneration for i in range(tp)]
    arrayTrainLoader = [aTrainLoader for i in range(tp)]
    arrayTestLoader = [aTestLoader for i in range(tp)]
    arrayValidationLoader = [aValidationLoader for i in range(tp)]
    arrayCat_df = [aCat_df for i in range(tp)]
    arrayBatch_size = [aBatch_size for i in range(tp)]
    arrayDevice = [aDevice for i in range(tp)]
    arrayCriterion = [aCriterion for i in range(tp)]
    cacheConfigClass = [copy.deepcopy(acacheConfigClass) for i in range(tp)]
    arrayMaxEpochsStop= [amax_epochs_stop for i in range(tp)]
    arrayNEpochs = [an_epochs for i in range(tp)]
    arrayCnnType = [acnnType for i in range(tp)]
    print('na hora de montar cacheConfigClass', cacheConfigClass)
    # print('arrayGeneration, arrayCriterion', arrayGeneration, arrayCriterion)
    return arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, cacheConfigClass, arrayMaxEpochsStop, arrayNEpochs, arrayCnnType

def calcFitness(generation, population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType):
    print('\n\n@@@@ Calculando fitness')
    tp = len(population)
    # print('calcFitness', trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
    arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayCacheConfigClass, arrayMaxEpochsStop, arrayNEpochs, arrayCnnType = saveGlobalVariables(generation, 
        trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, tp, cacheConfigClass, max_epochs_stop, n_epochs, cnnType)
    
    startAll = timeit.default_timer()
    iterations = [i for i in range(tp)]
    
    fitnessArray = []
    try:
        mp.set_start_method('spawn')
    except:
        print('error')
    
    print('after error', max_epochs_stop, n_epochs)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in zip(iterations, executor.map(calcFitnessIndividuo, population, iterations, arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayCacheConfigClass,  arrayMaxEpochsStop, arrayNEpochs, arrayCnnType)):
            iteration, (fitness, _) = result
            print('result', iteration, fitness)
            fitnessArray.append(fitness)

    endAll = timeit.default_timer()
    timeAll = endAll-startAll

    # fitnessArray = [calcFitnessIndividuo(population[i], i, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion) for i in range(tp)]
    return np.array(fitnessArray)
        
def calcFitnessIndividuo(individuo, i, generation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass, max_epochs_stop, n_epochs, cnnType, fileName=None):
    print('calcFitnessIndividuo = ', i, individuo, '\n')
    # print('\n pt1 ', i, trainLoader, testLoader, validationLoader)
    # print('\n pt2', cat_df, batch_size, device, criterion)
    print('cacheConfigClass', cacheConfigClass)
    print('device', device, 'maxepoch', max_epochs_stop)
    cacheValue = cacheConfigClass.verifyEntry(individuo)
    if cacheValue != None:
        print('\nachei cache', cacheValue, ' individuo = ', i, individuo, '\n fitness = ', cacheValue)
        return cacheValue, None

    model, optimizer = convertAgToCNN(individuo, device, cnnType)
    # print('epocas', epocas)

    if fileName != None:
        resultsPlotName = fileName
    else:
        resultsPlotName = 'runAG_geracao_' + str(generation) +'_individuo_'+str(i) 
    
    #treinamento
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, resultsPlotName, max_epochs_stop, n_epochs, device)
    
    history.to_csv('history_'+ resultsPlotName+ '.csv', index = False, header=True)

    allF1Score = history['validation_f1Score']
    
    # print('history', history)
    # the fitness is the f1-score of the validation set
    
    lastIndex = len(allF1Score) - 1
    agFitness = allF1Score[lastIndex]
    # print('fitness original', agFitness)
    
    isReducingLayerSize = verifyNetworkLayers(individuo) 

    if isReducingLayerSize == False:
        agFitness = agFitness*0.6
        # print('penalizada a rede', agFitness)
    # else:
        # print('rede nao penalizada')

    return agFitness, model
