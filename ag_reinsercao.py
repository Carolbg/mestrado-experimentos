import numpy as np

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

    tp = len(childPopulation)
    bestParent = np.array([bestParent])
    bestParentFitness = np.array([bestParentFitness])
    allPopulation = np.concatenate((bestParent, childPopulation))
    # print('allPopulation', allPopulation.shape)
    allFitness = np.concatenate((bestParentFitness, childrenFitness))
    # print('allFitness', allFitness.shape)
    sortedPopulation, sortedFitness = sortGenerationDecrescente(allPopulation, allFitness)
    newGeneration = sortedPopulation[0:tp]
    # print('newGeneration', newGeneration.shape)
    newGenerationFitness = sortedFitness[0:tp]
    # print('newGenerationFitness', newGenerationFitness.shape)
    
    return newGeneration, newGenerationFitness

def findBestIndividuo(population, populationFitness):
    sortedPopulation, sortedFitness = sortGenerationDecrescente(population, populationFitness)
    bestIndividuo = sortedPopulation[0]
    bestIndividuoFitness = sortedFitness[0]
    # print('bestIndividuo, bestIndividuoFitness', bestIndividuo, bestIndividuoFitness)
    return bestIndividuo, bestIndividuoFitness