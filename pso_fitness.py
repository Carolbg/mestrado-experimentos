import copy
import timeit
import numpy as np
import concurrent.futures
from prepareDataDictionary import getCommonArgs
import multiprocessing as mp
from training import train
from testing import evaluate
from pso_particleToCnn import *

def convertVariablesToArray(aGeneration, aTrainLoader, aTestLoader, aValidationLoader, aCat_df, aBatch_size, aDevice, aCriterion, tp, amax_epochs_stop, an_epochs, acnnType, acacheConfigClass):
    arrayGeneration = [aGeneration for i in range(tp)]
    arrayTrainLoader = [aTrainLoader for i in range(tp)]
    arrayTestLoader = [aTestLoader for i in range(tp)]
    arrayValidationLoader = [aValidationLoader for i in range(tp)]
    arrayCat_df = [aCat_df for i in range(tp)]
    arrayBatch_size = [aBatch_size for i in range(tp)]
    arrayDevice = [aDevice for i in range(tp)]
    arrayCriterion = [aCriterion for i in range(tp)]
    arrayCacheConfigClass = [copy.deepcopy(acacheConfigClass) for i in range(tp)]
    arrayMaxEpochsStop= [amax_epochs_stop for i in range(tp)]
    arrayNEpochs = [an_epochs for i in range(tp)]
    arrayCnnType = [acnnType for i in range(tp)]
    # print('na hora de montar cacheConfigClass', cacheConfigClass)

    # print('arrayGeneration, arrayCriterion', arrayGeneration, arrayCriterion)
    return arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayMaxEpochsStop, arrayNEpochs, arrayCnnType, arrayCacheConfigClass

def calcFitness(generation, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType, cacheConfigClass):
    print('\n\n@@@@ Calculando fitness')
    tp = len(swarm)
    # print('calcFitness', trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
    arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayMaxEpochsStop, arrayNEpochs, arrayCnnType, arrayCacheConfigClass = convertVariablesToArray(generation, 
        trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, tp, max_epochs_stop, n_epochs, cnnType, cacheConfigClass)
    
    startAll = timeit.default_timer()
    iterations = [i for i in range(tp)]
    
    fitnessArray = []
    try:
        mp.set_start_method('spawn')
    except:
        print('error')
    
    # print('after error', max_epochs_stop, n_epochs)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in zip(iterations, executor.map(calculateParticleFitness, swarm, iterations, arrayGeneration, arrayTrainLoader, arrayTestLoader, arrayValidationLoader, arrayCat_df, arrayBatch_size, arrayDevice, arrayCriterion, arrayMaxEpochsStop, arrayNEpochs, arrayCnnType, arrayCacheConfigClass)):
            iteration, (fitness, _) = result
            print('result', iteration, fitness)
            swarm[iteration]['positionFitness'] = fitness
            # print('swarm[iteration]',swarm[iteration])
            # fitnessArray.append(fitness)

    endAll = timeit.default_timer()
    timeAll = endAll-startAll

    # fitnessArray = [calcFitnessIndividuo(population[i], i, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion) for i in range(tp)]
    # return np.array(fitnessArray)

def verifyNetworkLayersParticle(particle):
    
    numberNeurons = []

    for i in range(1, len(particle)):
        #Considerando apenas as camadas densamente conectadas
        if particle[i]['layerType'] == 'FC':
            numberNeurons.append(particle[i]['layerNumber'])
            
    # print('numberNeurons', numberNeurons, flush=True)
    isReducingLayerSize = False
    
    if(all(numberNeurons[i] >= numberNeurons[i + 1] for i in range(len(numberNeurons)-1))): 
        isReducingLayerSize = True
    # print('isReducingLayerSize', isReducingLayerSize, flush=True)
    
    return isReducingLayerSize

def calculateParticleFitness(particle, i, generation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType, cacheConfigClass, fileName=None):
    # print('calculateParticleFitness = ', i, particle, '\n')
    # print('\n pt1 ', i, trainLoader, testLoader, validationLoader)
    # print('\n pt2', cat_df, batch_size, device, criterion)
    # print('device', device, 'maxepoch', max_epochs_stop)

    particlePosition = particle['position']

    cacheValue = cacheConfigClass.verifyEntry(particlePosition)
    if cacheValue != None:
        print('\n achei cache', cacheValue, ' individuo = ', i, particlePosition, '\n fitness = ', cacheValue)
        return cacheValue, None

    model, optimizer = convertParticleToCNN(particlePosition, device, cnnType)
    print('optimizer', optimizer)

    if fileName != None:
        resultsPlotName = fileName
    else:
        resultsPlotName = 'runPSO_geracao_' + str(generation) +'_individuo_'+str(i) 
    
    #treinamento
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, resultsPlotName, max_epochs_stop, n_epochs, device)
    
    history.to_csv('history_'+ resultsPlotName+ '.csv', index = False, header=True)

    allF1Score = history['validation_f1Score']
    
    # print('history', history)
    # the fitness is the f1-score of the validation set
    
    lastIndex = len(allF1Score) - 1
    fitness = allF1Score[lastIndex]
    # print('fitness original', fitness)
    
    isReducingLayerSize = verifyNetworkLayersParticle(particlePosition) 

    if isReducingLayerSize == False:
        fitness = fitness*0.6
    # print('fitness final', fitness)

    
    return fitness, model
