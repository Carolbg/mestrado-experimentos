import numpy as np
from ag_cacheConfig import parseIndividuoToKey

def sortGenerationDecrescente(allPopulation, allFitness):
    # print('allFitness', allFitness)
    sortedIndex = np.argsort(-1*allFitness)
    # print('sortedIndex', sortedIndex)
    sortedPopulation = allPopulation[sortedIndex]
    # print('sortedPopulation', sortedPopulation)
    sortedFitness = allFitness[sortedIndex]
    # print('sortedFitness', sortedFitness)

    return sortedPopulation, sortedFitness

def selectNewGenerationDecrescente(bestParent, bestParentFitness, childPopulation, childrenFitness):
    print('\n\n@@@@ Reinsercao')

    # print('bestParent, bestParentFitness', bestParent, bestParentFitness)
    # print(' childPopulation, childrenFitness', childPopulation, childrenFitness)

    tp = len(childPopulation)
    # print('tp', tp)
    bestParent = np.array([bestParent])
    bestParentFitness = np.array([bestParentFitness])
    allPopulation = np.concatenate((bestParent, childPopulation))
    # print('allPopulation', allPopulation)
    allFitness = np.concatenate((bestParentFitness, childrenFitness))
    # print('allFitness', allFitness)
    sortedPopulation, sortedFitness = sortGenerationDecrescente(allPopulation, allFitness)
    # print('sortedPopulation, sortedFitness', sortedPopulation, sortedFitness)
    newGeneration = sortedPopulation[0:tp]
    # print('newGeneration', newGeneration.shape)
    newGenerationFitness = sortedFitness[0:tp]
    # print('newGenerationFitness', newGenerationFitness.shape)
    
    return newGeneration, newGenerationFitness

def findBestIndividuo(population, populationFitness):
    sortedPopulation, sortedFitness = sortGenerationDecrescente(population, populationFitness)
    
    m = max(sortedFitness)
    indexMax = [i for i, j in enumerate(sortedFitness) if j == m]
    numberMax = len(indexMax)
    if numberMax == 1:
        bestIndividuo = sortedPopulation[0]
        bestIndividuoFitness = sortedFitness[0]
    else:
        print('\nmultiplos individiduos com maxValue = ',str(numberMax))
        #se tiver mais de um individuo com o maior valor, quero o que tem a menor cnn
        key = [parseIndividuoToKey(sortedPopulation[i]) for i in range(numberMax)]
        indexMinStr = key.index(min(key, key=len))
        bestIndividuo = sortedPopulation[indexMinStr]
        bestIndividuoFitness = sortedFitness[indexMinStr]
        
    # print('bestIndividuo, bestIndividuoFitness', bestIndividuo, bestIndividuoFitness)
    return bestIndividuo, bestIndividuoFitness