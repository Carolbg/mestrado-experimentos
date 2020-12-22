from ag_cnnFromAG import *
import ag_cacheConfig
from training import train
from testing import evaluate
# from cacheClass import cacheConfigClass
import copy
import timeit
import numpy as np
import concurrent.futures
import multiprocessing as mp

def saveGlobalVariables(aGeneration, aTrainLoader, aTestLoader, aValidationLoader, aCat_df, aBatch_size, aDevice, aCriterion, tp, acacheConfigClass):
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
    print('na hora de montar cacheConfigClass', cacheConfigClass)
    # print('arrayGeneration, arrayCriterion', arrayGeneration, arrayCriterion)
    return arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, cacheConfigClass

def calcFitness(generation, population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass):
    print('\n\n@@@@ Calculando fitness')
    tp = len(population)
    # print('calcFitness', trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
    arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayCacheConfigClass = saveGlobalVariables(generation, 
        trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, tp, cacheConfigClass)
    
    startAll = timeit.default_timer()
    iterations = [i for i in range(tp)]
    
    fitnessArray = []
    try:
        mp.set_start_method('spawn')
    except:
        print('error')
    
    print('after error')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in zip(iterations, executor.map(calcFitnessIndividuo, population, iterations, arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayCacheConfigClass)):
            # print('result', result)
            iteration, fitness = result
            fitnessArray.append(fitness)

    endAll = timeit.default_timer()
    timeAll = endAll-startAll

    # fitnessArray = [calcFitnessIndividuo(population[i], i, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion) for i in range(tp)]
    return np.array(fitnessArray)
        
def calcFitnessIndividuo(individuo, i, generation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, cacheConfigClass):
    # print('calcFitnessIndividuo = ', i, individuo, '\n')
    # print('\n pt1 ', i, trainLoader, testLoader, validationLoader)
    # print('\n pt2', cat_df, batch_size, device, criterion)
    print('cacheConfigClass', cacheConfigClass)
    cacheValue = cacheConfigClass.verifyEntry(individuo)
    if cacheValue != None:
        print('\nachei cache', cacheValue, ' individuo = ', i, individuo, '\n fitness = ', cacheValue)
        return cacheValue

    model, optimizer, epocas = convertAgToCNN(individuo, device)
    # print('epocas', epocas)
    resultsPlotName = 'runAG_geracao_' + str(generation) +'_individuo_'+str(i) 
    
    #treinamento
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, resultsPlotName, epocas, epocas, device)
    
    history.to_csv('history_'+ resultsPlotName+ '.csv', index = False, header=True)

    #teste
    historyTest, cmTest = evaluate(model, testLoader, criterion, 2, resultsPlotName, device)
    testAcc = historyTest['test_acc'][0]
    
    # print('@@@@ individuo = ', i, individuo, '\n fitness = ', testAcc)
    return testAcc