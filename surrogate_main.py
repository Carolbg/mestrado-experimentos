from randomForest import *
from surrogate_encoding import *
import numpy as np
from sklearn.model_selection import train_test_split 

def mainSurrogate(population, populationFitness):
    randomForestModel = createRandomForest(100)
    encodedPopulation = []
    # print('population', population)

    for individual in population:
        encodedIndividual = encodeAGIndividual(individual)
        npData = np.array(encodedIndividual)
        # print('encodedIndividual',npData.shape)
        
        flattenIndividual = npData.flatten()
        # print('flattenIndividual', flattenIndividual)
        # print('flattenIndividual', flattenIndividual.shape)
        encodedPopulation.append(flattenIndividual)

    # print('encodedPopulation', encodedPopulation)

    randomForestModel = trainModel(randomForestModel, encodedPopulation, populationFitness)
    return randomForestModel
    
def mainPSOSurrogate(swarm):
    randomForestModel = createRandomForest(100)
    encodedPopulation = []
    swarmFitness = []
    
    for particle in swarm:
        encodedParticle = encodeParticle(particle['position'])
        # print('size = len(encodedParticle)', len(encodedParticle))
        npData = np.array(encodedParticle)
        # print('encodedIndividual',npData.shape)
        
        flattenIndividual = npData.flatten()
        # # print('flattenIndividual', flattenIndividual)
        # print('flattenIndividual', flattenIndividual.shape)
        
        if len(flattenIndividual) > 0:
            encodedPopulation.append(flattenIndividual)
            swarmFitness.append(particle['positionFitness'])

        # add melhor local 
        encodedBestParticle = encodeParticle(particle['bestPosition'])
        npBestData = np.array(encodedBestParticle)
        flattenBestParticle = npBestData.flatten()
        if (flattenBestParticle == flattenIndividual).all():
            print('equals', flattenBestParticle, '\n', 'flattenIndividual', flattenIndividual)
            continue
        else:
            print('NOT equals', flattenBestParticle, '\n', 'flattenIndividual', flattenIndividual)

            if len(flattenBestParticle) > 0:
                encodedPopulation.append(flattenBestParticle)
                swarmFitness.append(particle['bestFitness'])

    # print('encodedPopulation', encodedPopulation)

    # add melhor global 
    encodedParticle = encodeParticle(particle['bestGlobalPosition'])
    npData = np.array(encodedParticle)
    
    flattenIndividual = npData.flatten()
    
    if len(flattenIndividual) > 0:
        encodedPopulation.append(flattenIndividual)
        swarmFitness.append(particle['bestGlobalFitness'])
    
    randomForestModel = trainModel(randomForestModel, encodedPopulation, swarmFitness)
    return randomForestModel
    