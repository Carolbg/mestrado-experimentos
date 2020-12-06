from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *
from ag_cnnInit import *
from ag_fitness import *
from ag_reinsercao import *
import ag_cacheConfig

def main(tp=10, tour=2, tr=80, numberIterations=10, tm=20, isNumpy=True):
    print('tp, tour, tr, numberIterations, tm, isNumpy', tp, tour, tr, numberIterations, tm, isNumpy)
    trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion = prepareCNN(isNumpy)
    ag_cacheConfig.initCache()
    sequenceIndividual = [i for i in range(12)]
    
    population = initializePopulation(tp)
    populationFitness = calcFitness(0, population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
    
    ag_cacheConfig.savePopulationToCache(population, populationFitness)
    # print('\ncacheStore = ', ag_cacheConfig.cacheStore)

    for i in range(numberIterations):
        print('\n\n $$$$$$$$ Geração ', i)
        print('population = ', population)
        print('populationFitness = ', populationFitness)
        bestParent, bestParentFitness = findBestIndividuo(population, populationFitness)
        print('bestParent, bestParentFitness', bestParent, bestParentFitness)
        selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
        newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)

        newPopulation = applyMutation(newPopulation, tm, tp)
        newPopulationFitness = calcFitness(i+1, newPopulation, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion)
        
        # print('\n\n Saving new generated items of geração ', i)
        ag_cacheConfig.savePopulationToCache(newPopulation, newPopulationFitness)

        population, populationFitness = selectNewGenerationDecrescente(bestParent, bestParentFitness, newPopulation, newPopulationFitness)
        
        # print('\ncacheStore = ', ag_cacheConfig.cacheStore)
    
    print('\nFinal population\n')
    print('population = ', population)
    print('populationFitness = ', populationFitness)
    bestParent, bestParentFitness = findBestIndividuo(population, populationFitness)
    print('bestParent, bestParentFitness', bestParent, bestParentFitness)

    return population, populationFitness

main()