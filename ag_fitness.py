from ag_cnnFromAG import *
from ag_cacheConfig import *
from training import train
from testing import evaluate

import timeit
import numpy as np
import concurrent.futures
import multiprocessing as mp

def saveGlobalVariables(aTrainLoader, aTestLoader, aValidationLoader, aCat_df, aBatch_size, aDevice, aCriterion):
    global trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion
    
    trainLoader = aTrainLoader
    testLoader = aTestLoader
    validationLoader = aValidationLoader
    cat_df = aCat_df
    batch_size = aBatch_size
    device = aDevice
    criterion = aCriterion

def calcFitness(population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion):
    print('\n\n@@@@ Calculando fitness')
    tp = len(population)
    # print('calcFitness', trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
    saveGlobalVariables(trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
    
    startAll = timeit.default_timer()
    iterations = [i for i in range(tp)]
    
    fitnessArray = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in zip(iterations, executor.map(calcFitnessIndividuo, population, iterations)):
            print('result', result)
            iteration, fitness = result
            fitnessArray.append(fitness)

    endAll = timeit.default_timer()
    timeAll = endAll-startAll

    # fitnessArray = [calcFitnessIndividuo(population[i], i, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion) for i in range(tp)]
    return np.array(fitnessArray)
        
def calcFitnessIndividuo(individuo, i):
    # print('calcFitnessIndividuo = ', i, individuo, '\n')
    # print('\n pt1 ', i, trainLoader, testLoader, validationLoader)
    # print('\n pt2', cat_df, batch_size, device, criterion)

    cacheValue=verifyEntry(individuo)
    if cacheValue != None:
        print('achei cache', cacheValue)
        return cacheValue

    model, optimizer, epocas = convertAgToCNN(individuo, device)
    # print('epocas', epocas)
    resultsPlotName = 'runAG_individuo_'+str(i)
    
    #treinamento
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, resultsPlotName, epocas, epocas, device)

    #teste
    historyTest, cmTest = evaluate(model, testLoader, criterion, 2, resultsPlotName, device)
    testAcc = historyTest['test_acc'][0]  
    
    addNewEntry(individuo, testAcc)
    
    print('@@@@ individuo = ', i, individuo, '\n fitness = ', testAcc)
    return testAcc