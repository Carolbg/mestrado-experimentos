from ag_initialize import *
from surrogate_encoding import *
from surrogate_main import *
import numpy as np
from pso_initialize import *
from surrogate_fitness import *
from psoCacheClass import PSOCacheClass

# individual = initializeIndividual()
# print('individual', individual)
# encodedIndividual = encodeAGIndividual(individual)
# print('encodedIndividual', encodedIndividual)
# print('\n\n')

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



#PSO

swarm = initializeSwarm(10)
# print('\n swarm[0]', swarm[0], '\n')
# print('\n swarm[1]', swarm[1], '\n')
# print('\n swarm[2]', swarm[2], '\n')
# print('\n swarm[3]', swarm[3], '\n')

swarm[0]['positionFitness'] = 100
swarm[1]['positionFitness'] = 80
swarm[2]['positionFitness'] = 99
swarm[3]['positionFitness'] = 98
swarm[4]['positionFitness'] = 87

randomTreeModel = mainPSOSurrogate(swarm)

cacheConfigClass = PSOCacheClass()
calcSurrogatePSOFitness(randomTreeModel, swarm, cacheConfigClass, 5, 5, 5)

# print('\n resultado \n')
# print('\n swarm[5]', swarm[5], '\n')
# print('\n swarm[6]', swarm[6], '\n')
# print('\n swarm[7]', swarm[7], '\n')
# print('\n swarm[8]', swarm[8], '\n')
# print('\n swarm[9]', swarm[9], '\n')

# for particle in swarm:
#     encodedParticle = encodeParticle(particle)
#     print('@ encodedParticle', encodedParticle)
