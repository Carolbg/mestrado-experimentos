from ag_initialize import *
from surrogate_encoding import *
from surrogate_main import *
import numpy as np

individual = initializeIndividual()
print('individual', individual)
encodedIndividual = encodeAGIndividual(individual)

# individual = initializeIndividual()
# encodedIndividual = encodeAGIndividual(individual)
# print('encodedIndividual', encodedIndividual)
# npData = np.array(encodedIndividual)
# flattenIndividual = npData.flatten()
# print('flattenIndividual', flattenIndividual)        

# population = initializePopulation(3)
# populationFitness = [6,8,10]
# mainSurrogate(population, populationFitness)

# from randomForest import *
# from surrogate_encoding import *
# import numpy as np
# from sklearn.model_selection import train_test_split 


# x_train, x_test, y_train, y_test = train_test_split(encodedPopulation, populationFitness, test_size = 0.2, random_state = 1)
# randomForestModel = trainModel(randomForestModel, x_train, y_train)
# ypred = testModel(randomForestModel, x_test)
# calcRMSE(y_test, ypred)
