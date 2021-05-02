import numpy as np
import math
import random
from pso_initialize import *
from pso_fitness import *
from utils_readAllData import *

def printSwarm(swarm):
    for i in range(1, len(swarm)):
        print(swarm[i], '\n')

def updateBestSolutionParticle(particle):
    if particle['positionFitness'] > particle['bestFitness']:
        particle['bestFitness'] = particle['positionFitness']
        particle['bestPosition'] = particle['position']
    return particle

def bestNeighbourPosition(swarm, populationSize):
    bestSolutionCost = swarm[0]['bestFitness']
    bestSolution = swarm[0]['bestPosition']
    # print('troquei i = 0')

    #Encontrar a melhor solucao entre todos os vizinhos
    for i in range(1, populationSize):
        if swarm[i]['bestFitness'] > bestSolutionCost:
            bestSolutionCost = swarm[i]['bestFitness']
            bestSolution = swarm[i]['bestPosition']
            # print('troquei i = ', i)
    
    #Atualizar todo mundo com o melhor vizinho
    for i in range(0, populationSize):
        swarm[i]['bestGlobalFitness'] = bestSolutionCost
        swarm[i]['bestGlobalPosition'] = bestSolution
    return swarm

def calcDiffLayers(layerP1, layerP2):
    # Commenting this to make diff equals to Franciso paper
    # if layerP1['layerType'] == 'LR':
    #     return { 'layerType': 'LR',
    #      'layerNumber': (layerP1['layerNumber']-layerP2['layerNumber']) 
    #     }

    if layerP1['layerType'] == layerP2['layerType']:
        return { 'layerType': None }
    return layerP1

def calcDiffTwoParticles(particle1, particle2):
    # calc diff particle1 - particle2
    diff = []
    sizeP1 = len(particle1)
    sizeP2 = len(particle2)
    maxSize = max(sizeP1, sizeP2)

    for i in range(maxSize):
        # print('i', i)
        if i < sizeP1 and i < sizeP2:
            diffLayer = calcDiffLayers(particle1[i], particle2[i])
            # print('if1 diffLayer = ', diffLayer)
            diff.append(diffLayer)
        elif i < sizeP1:
            # print('if2')
            diff.append(particle1[i])
        else:
            # print('else')
            layer = { 'layerType': -1 }
            diff.append(layer)

    return diff

def calcVelocity(Cg, diffPBest, diffGBest, pBest, gBest):
    velocity = []
    sizeDiffPBest = len(diffPBest)
    sizeDiffGBest = len(diffGBest)
    # print('sizeDiffPBest', sizeDiffPBest, 'sizeDiffGBest', sizeDiffGBest)

    maxSize = max(sizeDiffPBest, sizeDiffGBest)

    #verificando caso especial onde as diffs sao iguais a 0
    if sizeDiffPBest == sizeDiffGBest:
        isNonePBest = [item['layerType'] == None for item in diffPBest]
        isNoneGBest = [item['layerType'] == None for item in diffGBest]
        if all(isNoneGBest) and all(isNonePBest):
            # print('using the actual values')
            diffPBest = pBest
            diffGBest = gBest
        # else:
        #     print('same size has something different from Non')

    for i in range(maxSize):
        randValue = random.random()
        # print('randValue', randValue)
        if randValue < Cg:
            # print('using gBest')
            if i < sizeDiffGBest:
                # print('if gbest')
                velocity.append(diffGBest[i])
            else:
                # print('else gbest')
                velocity.append( { 'layerType': None })
        else:
            # print('using pBest')
            if i < sizeDiffPBest:
                # print('if pBest')
                velocity.append(diffPBest[i])
            else:
                # print('else pBest')
                velocity.append( { 'layerType': None })
        # print('\n')
    # print('velocity', velocity)
    return velocity

def updateParticlePosition(particle, newVelocity):
    sizeParticle = len(particle)
    sizeVelocity = len(newVelocity)
    # print('sizeParticle', sizeParticle, 'sizeVelocity', sizeVelocity)
    maxSize = max(sizeParticle, sizeVelocity)
    # print('maxSize', maxSize)
    
    newPosition = []
    
    for i in range(sizeVelocity):
        # print('i', i, "newVelocity[i]['layerType']", newVelocity[i]['layerType'])
        if newVelocity[i]['layerType'] == None and i < sizeParticle:
            # print('if 1')
            newPosition.append(particle[i])
            # print('newPosition', newPosition)
        elif newVelocity[i]['layerType'] == 'FC' or newVelocity[i]['layerType'] == 'Dropout' or newVelocity[i]['layerType'] == 'LR':
            # print('if 2')
            newPosition.append(newVelocity[i])
            # print('newPosition', newPosition)
        # else:
        #     print('no else')
        # print('\n')

    return newPosition

def validateParticle(particle):
    validParticle = []
    validParticle.append(particle[0])
    particleSize = len(particle)
    i = 1
    j = i+1
    while i < particleSize:
        if j < particleSize:
            if particle[i]['layerType'] == 'Dropout' and particle[j]['layerType'] == 'Dropout':
                j = j+1
            else:
                validParticle.append(particle[i])
                i = j 
                j = j+1
        else:
            if i+1 < j:
                validParticle.append(particle[i])
                break
            else:
                validParticle.append(particle[i])
                i = i+1
    
    return validParticle

def PSO(iterations=10, populationSize=10, Cg=0.5, isNumpy=False, cnnType=1, nEpochs=30):
    # If running on colab keep the next line commented
    readData(isNumpy, nEpochs)

    trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()

    swarm = initializeSwarm(populationSize)
    calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType)
    for particle in swarm:
        updateBestSolutionParticle(particle)
        validateParticle(particle['position'])
    
    swarm = bestNeighbourPosition(swarm, populationSize)
    # print('swarm', swarm)
    # print('\n')
    # #printSwarm(swarm)
    for iteration in range(iterations):
        print('\n****** iteration', iteration, '******\n')
        
        # updates gbest (best particle of the population)
        # printSwarm(swarm)
        for particle in swarm:
            # Cac diff pBest - p
            diffPBest = calcDiffTwoParticles(particle['bestPosition'], particle['position'])
        
            # Cac diff gBest - p
            diffGBest = calcDiffTwoParticles(particle['bestGlobalPosition'], particle['position'])

            #Calc velocity
            newVelocity = calcVelocity(Cg, diffPBest, diffGBest, particle['bestPosition'], particle['bestGlobalPosition'])
            
            # print("particle['position']", particle['position'])
            # print('newVelocity', newVelocity)
            particle['position'] = updateParticlePosition(particle['position'], newVelocity)
            # print("BEFORE: particle['position']", particle['position'])
            particle['position'] = validateParticle(particle['position'])
            # print("AFTER: particle['position']", particle['position'])
            # print('\n')
        
        # print('swarm', swarm)
        # print('\n')
        
        #calc das redes novas geradas com o update das posicoes
        calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType)
        # print('finish fitness')
        for particle in swarm:
            updateBestSolutionParticle(particle)
        swarm = bestNeighbourPosition(swarm, populationSize)
        print('\n')
    
    return swarm 
