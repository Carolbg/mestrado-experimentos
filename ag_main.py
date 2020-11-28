from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *

def main(tp=10, tour=3, tr=80, numberIterations=10, tm=10):
    sequenceIndividual = [i for i in range(12)]

    population = initializePopulation(tp)
    populationFitness = [6,8,10,4,5,9]

    for i in range(numberIterations):
        print('Geração ', i)
        selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
        newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)
        newPopulation = applyMutation(newPopulation, tm, tp)

        #do elitismo
        population = newPopulation
    return population
    