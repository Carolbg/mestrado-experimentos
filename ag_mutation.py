from random import sample, random, uniform, seed
import math
from ag_utils import *

def applyMutation(childPopulation, tm, tp):
    sequenceChild = [i for i in range(len(childPopulation))]
    
    numberChildrenToMutate = int(tm * tp /100)
    individualsToMutate = sample(sequenceChild, numberChildrenToMutate)
    # print('Individuos para mutacao', individualsToMutate)
    
    mutatedIndividuals = [applyMutationInd(childPopulation[i], sequenceChild) for i in individualsToMutate]

    childPopulation[individualsToMutate] = mutatedIndividuals

    return childPopulation


def applyMutationInd(individuo, sequenceChild):
    # print('#### before individuo', individuo)
    geneToMutate = randomInt(0, 11)
    # print('geneToMutate', geneToMutate)

    subGenesIndex = randomInt(0, 1)
    # print('subGenesIndex', subGenesIndex)

    #Gene to mutate:
    initialSeed = uniform(0, 5000)
    seed(initialSeed)
    
    individuo[geneToMutate] = mutateGene(individuo, geneToMutate, subGenesIndex)
    # print('####  after individuo', individuo)
    
    return individuo

#o subGene eh pra saber se aplico no presente ou ausente ou no range
def mutateGene(individuo, geneIndex, subGene):
    geneToMutate = individuo[geneIndex]
    # print('@@@ original', geneToMutate, 'geneIndex', geneIndex)
    if geneIndex >= 2:
        newValue= mutateGeneLayers(geneToMutate, geneIndex, subGene)
        # print('@@@ mutated', newValue)
        return newValue
    else:
        newValue = randomInt(1, 6)

        if newValue == geneToMutate:
            newValue = randomInt(1, 6)
        
        if newValue == geneToMutate:
            print('continua igual')
        newValue = [newValue]

    # print('@@@ mutated', newValue)

    return newValue
    
def mutateGeneLayers(originalValue, index, subGene):
    # print('seed', initialSeed)
    if index % 2 == 0:
        return mutateDenseLayers(originalValue, index, subGene)
    else:
        return mutateDropoutLayers(originalValue, index, subGene)

def mutateDenseLayers(originalValue, index, subGene):
    # print('mutateDenseLayers: subGene = ', subGene)
    # print('before originalValue', originalValue)
    if subGene == 0:
        newValue = 1 if originalValue[subGene] == 0 else 0
    else:
        newValue = randomInt(3, 12)
        if originalValue[subGene] == newValue:
            newValue = randomInt(3, 12)

        if originalValue[subGene] == newValue:
            print('continua igual')

    # print('newValue', newValue)
    originalValue[subGene] = newValue
    # print('mutateDenseLayers: after originalValue', originalValue)

    return originalValue

def mutateDropoutLayers(originalValue, index, subGene):
    # print('mutateDropoutLayers: subGene = ', subGene)
    # print('before originalValue', originalValue)
    if subGene == 0:
        newValue = 1 if originalValue[subGene] == 0 else 0
    else:
        newValue = randomFloat(0, 0.6)
        if originalValue[subGene] == newValue:
            newValue = randomFloat(0, 0.6)

        if originalValue[subGene] == newValue:
            print('continua igual')

    # print('newValue', newValue)
    originalValue[subGene] = newValue
    # print('mutateDropoutLayers: after originalValue', originalValue)

    return originalValue
    