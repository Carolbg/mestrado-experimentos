from pso_initialize import *
from pso_fitness import *
from utils_readAllData import *
from pso import *

def singleIteration(swarm, populationSize, Cg):
# updates gbest (best particle of the population)
    swarm = bestNeighbourPosition(swarm, populationSize)
    #     #printSwarm(swarm)
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
        particle['position'] = validateParticle(particle['position'])
        # print("particle['position']", particle['position'])
        # print('\n')
    return swarm