from ag_initialize import *
from ag_selection import *
from ag_crossover import *

def main(tp=10, tour=3, tr=80, numberIterations=10):
    sequenceIndividual = [i for i in range(12)]

    population = initializePopulation(tp)
    populationFitness = [6,8,10,4,5,9]

    for i in range(numberIterations):
        selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
        newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)
        #do elitismo
        population = newPopulation
    return population
    