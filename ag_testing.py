from ag_initialize import *
from ag_selection import *
from ag_crossover import *
from ag_mutation import *

tp=4
tour=3
tr=80
tm=50
sequenceIndividual = [i for i in range(12)]
population = initializePopulation(tp)

populationFitness = [6,8,10,4]#,5,9]
selectedParents1, selectedParents2 = selectParentsWithTorneio(population, populationFitness, tour)
newPopulation = applyCrossover(selectedParents1, selectedParents2, tr, sequenceIndividual)

newPopulationAfterMutation = applyMutation(newPopulation, tm, tp)
# childPopulation=newPopulation#.copy()
# sequenceChild = [i for i in range(len(childPopulation))]
    
# numberChildrenToMutate = int(tm * tp /100)
# individualsToMutate = sample(sequenceChild, numberChildrenToMutate)
# print('Individuos para mutacao', individualsToMutate)

# mutatedIndividuals = [applyMutationInd(childPopulation[i], sequenceChild) for i in individualsToMutate]
