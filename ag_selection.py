import numpy as np
from random import sample, uniform, seed, randint, randrange

def selectParentsWithTorneio(population, populationFitness, tour):
    print('\n\n@@@@ Selecao')
    numberChildren = len(population)
    numberParents = int(numberChildren/2)

    parents1 = np.zeros((numberParents, 11), dtype=object)
    parents2 = np.zeros((numberParents, 11), dtype=object)

    sequence = [i for i in range(len(populationFitness))]
    # print('sequence', sequence)
    
    for i in range(numberParents):
        indexParent1 = torneio(populationFitness, tour, sequence)
        indexParent2 = torneio(populationFitness, tour, sequence)
        print('indexParent1', indexParent1)
        print('indexParent2', indexParent2, '\n')

        parents1[i] = population[indexParent1]
        # print('parents1[i]', parents1[i])
        parents2[i] = population[indexParent2]
        # print('parents2[i]', parents2[i])
        
    return parents1, parents2

def torneio(populationFitness, tour, sequence):
    
    subgroupIndex = sample(sequence, tour)
    # print('subgroupIndex', subgroupIndex)
    
    subgroupFitness=[populationFitness[subgroupIndex[i]] for i in range(tour)]
    # print('subgroupFitness', subgroupFitness)
    # subgroupFitness = np.zeros(tour)
    
    indexMaxValue = np.argmax(subgroupFitness)
    # print('indexMaxValue', indexMaxValue)
    indexParent = subgroupIndex[indexMaxValue]
    # print('indexParent', indexParent)
    return indexParent
