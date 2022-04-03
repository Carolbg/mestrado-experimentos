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
        print('flattenIndividual', flattenIndividual.shape)
        encodedPopulation.append(flattenIndividual)

    print('encodedPopulation', encodedPopulation)

    randomForestModel = trainModel(randomForestModel, encodedPopulation, populationFitness)
    return randomForestModel
    