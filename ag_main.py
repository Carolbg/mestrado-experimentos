from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *
from ag_cnnInit import *
from ag_fitness import *
from ag_cacheConfig import *
from ag_reinsercao import *

def main(tp=10, tour=2, tr=80, numberIterations=10, tm=20, isNumpy=True):
    trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion = prepareCNN(isNumpy)
    initCache()
    sequenceIndividual = [i for i in range(12)]

    population = initializePopulation(tp)
    populationFitness = calcFitness(population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)

    for i in range(numberIterations):
        print('Geração ', i)
        bestParent, bestParentFitness = findBestIndividuo(population, populationFitness)

        selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
        newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)

        newPopulation = applyMutation(newPopulation, tm, tp)
        newPopulationFitness = calcFitness(newPopulation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)

        population, populationFitness = selectNewGenerationDecrescente(bestParent, bestParentFitness, newPopulation, newPopulationFitness)
        
    return population
    