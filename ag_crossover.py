from random import sample, uniform, seed
import numpy as np
from ag_utils import *
import copy

def crossover2points(parent1, parent2, tr, sequence):
    #tr = 80% entao vira 0.8
    # print('parent1', parent1)
    # print('\n parent2', parent2)
    # initialSeed = uniform(0, 5000)
    # seed(initialSeed)

    tr = tr/100
    randomNumber = uniform(0, 1)
    print('randomNumber', randomNumber)
    if randomNumber > tr:
        print('not changing individuals')
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    crossoverPoints = sample(sequence, 2)
    # print('crossoverPoints', crossoverPoints)

    if crossoverPoints[0] > crossoverPoints[1]:
        startPoint = crossoverPoints[1] 
        endPoint = crossoverPoints[0] 
    else:
        startPoint = crossoverPoints[0] 
        endPoint = crossoverPoints[1] 

    # print('parent1', parent1, '\parent2', parent2)
    # print('startPoint', startPoint, 'endPoint', endPoint)

    # child1 = parent1.copy()
    # child2 = parent2.copy()
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    child1[startPoint:endPoint]=parent2[startPoint:endPoint]
    child2[startPoint:endPoint]=parent1[startPoint:endPoint]
    print('child1', child1, '\nchild2', child2)

    return child1, child2

def crossover1point(parent1, parent2, tr, sequence):
    #tr = 80% entao vira 0.8
    # print('\n\n CROSSOVER 1 ponto')
    # print('parent1', parent1)
    # print('\n parent2', parent2)
    # initialSeed = uniform(0, 5000)
    # seed(initialSeed)

    tr = tr/100
    randomNumber = uniform(0, 1)
    # print('randomNumber', randomNumber)
    if randomNumber > tr:
        # print('not changing individuals')
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    crossoverPoint = randomInt(1, 9)
    
    # print('crossoverPoint', crossoverPoint)

    # child1 = parent1.copy()
    # child2 = parent2.copy()
     
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    child1[crossoverPoint:] = parent2[crossoverPoint:]
    child2[crossoverPoint:] = parent1[crossoverPoint:]
    
    # print('child1', child1, '\nchild2', child2)

    return child1, child2

def applyCrossover(parents1, parents2, tr, sequence):
    
    numberChildren = len(parents1)
    # print('numberChildren', numberChildren)
    print('\n\n@@@@ Crossover 1 ponto')
    children1, children2 =  map(list,zip(*[crossover1point(parents1[i], parents2[i], tr, sequence) for i in range(numberChildren)]))
    
    # print('\n\n@@@@ Crossover 2 pontos')
    # children1, children2 =  map(list,zip(*[crossover2points(parents1[i], parents2[i], tr, sequence) for i in range(numberChildren)]))

    # print('children1[0]', children1[0])
    childrenPopulation = np.concatenate((children1, children2))
    #  childrenPopulation = np.concatenate((np.array(children1), np.array(children2)))

    #     return childrenPopulation.toList()
    print('childrenPopulation', len(childrenPopulation))
    return childrenPopulation
