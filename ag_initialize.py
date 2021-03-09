from random import sample, uniform, seed, randint
import numpy as np
from ag_utils import *

def initializePopulation(tp):
    print('\n\n@@@@ Init pop')
    population = [initializeIndividual() for i in range(tp)]
    return np.array(population)

def initializeIndividual():
    individual = [initializeGene(i) for i in range(11)]
    # print('individual', len(individual))
    return np.array(individual)

def initializeGene(i):
    if i >= 1:
        return initializeGeneLayers(i)
    else:
        value = randomInt(1, 6)
        # print('value = ', value)
        return [value]
        
    # elif i == 0:
    #     lr = randomInt(1, 6)
    #     # L=10^ (−k1), k1 ∈ [1,6]
    #     # print('lr = ', lr)
    #     return [lr]
    # else:
    #     # Ne ∈ [10, 60]
    #     # 1 = 10, 2=20, 3=30, 4=40, 5=50, 6=60

    #     epocas = randomInt(1, 6)
    #     # print('epocas = ', epocas)
    #     return [epocas]

def initializeGeneLayers(i):
    initialSeed = uniform(0, 5000)
    seed(initialSeed)
    # print('seed', initialSeed)
    if i % 2 != 0:
        denseLayer = initializeDenseLayers(i)
        # print('denseLayer', denseLayer)
        return denseLayer
    else:
        dropoutLayer = initializeDropoutLayers(i)
        # print('dropoutLayer', dropoutLayer)
        return dropoutLayer

def initializeDenseLayers(i):
    isPresent = randomInt(0, 1)
    nroNeuronios = randomInt(3, 12)

    isPresent = randomInt(0, 1)
    nroNeuronios = randomInt(3, 12)

    return [isPresent, nroNeuronios]

def initializeDropoutLayers(i):
    isPresent = randomInt(0, 1)
    dropoutRate = randomFloat(0, 0.6)
    return [isPresent, dropoutRate]
