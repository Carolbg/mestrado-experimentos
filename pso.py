import numpy as np
import math
import random
from pso_initialize import *
from pso_fitness import *
from utils_readAllData import *

def calculateSolutionDistance(permutation,numberVertex, graph):
    distance = 0
    for j in range(numberVertex):
        pos1 = permutation[j]
        pos2 = permutation[(j+1) % numberVertex]
        distance = distance + graph[pos1, pos2]
    return distance

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

    #Encontrar a melhor solucao entre todos os vizinhos
    for i in range(1, populationSize):
        if swarm[i]['bestFitness'] > bestSolutionCost:
            bestSolutionCost = swarm[i]['bestFitness']
            bestSolution = swarm[i]['bestPosition']
    
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
    print('sizeDiffPBest', sizeDiffPBest, 'sizeDiffGBest', sizeDiffGBest)

    maxSize = max(sizeDiffPBest, sizeDiffGBest)

    #verificando caso especial onde as diffs sao iguais a 0
    if sizeDiffPBest and sizeDiffGBest:
        isNonePBest = [item['layerType'] == None for item in diffPBest]
        isNoneGBest = [item['layerType'] == None for item in diffGBest]
        if isNoneGBest and isNonePBest == 0:
            print('using the actual values')
            diffPBest = pBest
            diffGBest = gBest
        else:
            print('same size has something different from Non')

    
    for i in range(maxSize):
        randValue = random.random()
        print('randValue', randValue)
        if randValue < Cg:
            print('using gBest')
            if i < sizeDiffGBest:
                print('if gbest')
                velocity.append(diffGBest[i])
            else:
                print('else gbest')
                velocity.append( { 'layerType': None })
        else:
            print('using pBest')
            if i < sizeDiffPBest:
                print('if pBest')
                velocity.append(diffPBest[i])
            else:
                print('else pBest')
                velocity.append( { 'layerType': None })
        print('\n')
    print('velocity', velocity)
    return velocity

def updateVelocity(particle, tempVelocityP,tempVelocityG):
    #print('1 - tempVelocityP', tempVelocityP)
    tempVelocityP.extend(tempVelocityG)
    #print('2 - tempVelocityP', tempVelocityP)
    particle['velocity'] = tempVelocityP
    return particle

def calculateNewPosition(particle, tempVelocity, numberVertex, graph):
    # generates new solution for particle
    solution_particle = particle['position'].copy()
    for swap_operator in tempVelocity:
        if random.random() <= swap_operator[2]:
            # makes the swap
            aux = solution_particle[swap_operator[0]]
            solution_particle[swap_operator[0]] = solution_particle[swap_operator[1]]
            solution_particle[swap_operator[1]] = aux
    particle['position'] = solution_particle
    particle['positionFitness'] = calculateSolutionDistance(solution_particle, numberVertex, graph)

def PSO(iterations=10, populationSize=10, Cg=0.5, isNumpy=False, cnnType=1, nEpochs=30):
    # If running on colab keep the next line commented
    readData(isNumpy, nEpochs)

    trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()

    swarm = initializeSwarm(populationSize)
    calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType)
    for particle in swarm:
        updateBestSolutionParticle(particle)
        
    # #printSwarm(swarm)
    for iteration in range(iterations):
        print('\n****** iteration', iteration, '******')
        # updates gbest (best particle of the population)
        swarm = bestNeighbourPosition(swarm, populationSize)
    #     #printSwarm(swarm)
        for particle in swarm:
            # Cac diff pBest - p
            diffPBest = calcDiffTwoParticles(particle['bestPosition'], particle['position'])
            print("particle['bestPosition']", particle['bestPosition'])
            print("\n  particle['position']", particle['position'])
            print("\n diffPBest", diffPBest)
            
            # Cac diff gBest - p
            diffGBest = calcDiffTwoParticles(particle['bestGlobalPosition'], particle['position'])

            print("\n\n particle['bestGlobalPosition']", particle['bestGlobalPosition'])
            print("\n  particle['position']", particle['position'])
            print("\n diffGBest", diffGBest)
            input('stopping after diff')

            newVelocity = calcVelocity(Cg, diffPBest, diffGBest, particle['bestPosition'], particle['bestGlobalPosition'])
            input('stopping afterVelocity diff')
    #         (tempVelocityP, solution_pbest) = termWithAlpha(particle, numberVertex, alpha)
    #         (tempVelocityG, solution_gbest) = termWithBeta(particle, numberVertex, beta)
    #         updateVelocity(particle, tempVelocityP, tempVelocityG)
    #         tempVelocity = particle['velocity']
    #         calculateNewPosition(particle, tempVelocity, numberVertex, graph)
    #         updateBestSolutionParticle(particle)
    #     #printSwarm(swarm)
    #     #print(swarm[0])
    # swarm = bestNeighbourPosition(swarm, populationSize)
    return swarm 

def findBest(swarm, citiesName):
    bestCost = swarm[0]['bestGlobalFitness']
    bestPermutation = swarm[0]['bestGlobalPosition']
    cities = []
    for i in range (len(bestPermutation)):
        cities.append(citiesName[bestPermutation[i]])
    print('\n\nMenor caminho encontrado tem custo', bestCost )
    print('Caminho = ', cities )
    return (bestCost, bestPermutation, cities)
