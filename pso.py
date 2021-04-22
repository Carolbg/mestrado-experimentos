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
    if particle['positionFitness'] < particle['bestFitness']:
        particle['bestFitness'] = particle['positionFitness']
        particle['bestPosition'] = particle['position']
    return particle

def bestNeighbourSolution(swarm, populationSize):
    bestSolutionCost = swarm[0]['bestFitness']
    bestSolution = swarm[0]['bestPosition']

    #Encontrar a melhor solucao entre todos os vizinhos
    for i in range(1, populationSize):
        if swarm[i]['bestFitness'] < bestSolutionCost:
            bestSolutionCost = swarm[i]['bestFitness']
            bestSolution = swarm[i]['bestPosition']
    
    #Atualizar todo mundo com o melhor vizinho
    for i in range(0, populationSize):
        swarm[i]['bestNeighbourSolution'] = bestSolutionCost
        swarm[i]['bestNeighbourSolution'] = bestSolution
    return swarm

def termWithBeta(particle, numberVertex, beta):
    # generates all swap operators to calculate (gbest - x(t-1))
    tempVelocity = []
    solution_gbest = particle['bestNeighbourSolution'].copy()
    solution_particle = particle['position'].copy()
    for i in range(numberVertex):
        if solution_particle[i] != solution_gbest[i]:
            swap_operator = (i, solution_gbest.index(solution_particle[i]), beta)
            
            # append swap operator in the list of velocity
            tempVelocity.append(swap_operator)

            # makes the swap
            aux = solution_gbest[swap_operator[0]]
            solution_gbest[swap_operator[0]] = solution_gbest[swap_operator[1]]
            solution_gbest[swap_operator[1]] = aux
    return (tempVelocity, solution_gbest)

def updateVelocity(particle, tempVelocityP,tempVelocityG):
    #print('1 - tempVelocityP', tempVelocityP)
    tempVelocityP.extend(tempVelocityG)
    #print('2 - tempVelocityP', tempVelocityP)
    particle['velocity'] = tempVelocityP
    return particle

def termWithAlpha(particle, numberVertex, alpha):
# generates all swap operators to calculate (pbest - x(t-1))
    tempVelocity = []
    solution_pbest = particle['bestPosition'].copy()
    solution_particle = particle['position'].copy()
    for i in range(numberVertex):
        if solution_particle[i] != solution_pbest[i]:
            swap_operator = (i, solution_pbest.index(solution_particle[i]), alpha)
            
            # append swap operator in the list of velocity
            tempVelocity.append(swap_operator)

            # makes the swap
            aux = solution_pbest[swap_operator[0]]
            solution_pbest[swap_operator[0]] = solution_pbest[swap_operator[1]]
            solution_pbest[swap_operator[1]] = aux
    return (tempVelocity, solution_pbest)

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

def PSO(iterations=10, populationSize=10, Cg=0.7, isNumpy=False, cnnType=1, nEpochs=30):
    # If running on colab keep the next line commented
    readData(isNumpy, nEpochs)
    trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs = getData()

    swarm = initializeSwarm(populationSize)
    calcFitness(0, swarm, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion, max_epochs_stop, n_epochs, cnnType)
    print('swarm no init', swarm)

    # #printSwarm(swarm)
    # for iteration in range(iterations):
    #     #print('\n****** iteration', iteration, '******')
    #     # updates gbest (best particle of the population)
    #     swarm = bestNeighbourSolution(swarm, populationSize)
    #     #printSwarm(swarm)
    #     for particle in swarm:
    #         (tempVelocityP, solution_pbest) = termWithAlpha(particle, numberVertex, alpha)
    #         (tempVelocityG, solution_gbest) = termWithBeta(particle, numberVertex, beta)
    #         updateVelocity(particle, tempVelocityP, tempVelocityG)
    #         tempVelocity = particle['velocity']
    #         calculateNewPosition(particle, tempVelocity, numberVertex, graph)
    #         updateBestSolutionParticle(particle)
    #     #printSwarm(swarm)
    #     #print(swarm[0])
    # swarm = bestNeighbourSolution(swarm, populationSize)
    return swarm 

def findBest(swarm, citiesName):
    bestCost = swarm[0]['bestNeighbourSolution']
    bestPermutation = swarm[0]['bestNeighbourSolution']
    cities = []
    for i in range (len(bestPermutation)):
        cities.append(citiesName[bestPermutation[i]])
    print('\n\nMenor caminho encontrado tem custo', bestCost )
    print('Caminho = ', cities )
    return (bestCost, bestPermutation, cities)
