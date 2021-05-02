import random

# export interface Layer {
#     typeLayer: 'FC' | 'D' | 'LR',
#     value: number
# }}
# initialPosition: Layer[]
# initialPosition = arquitetura de cnn

def createParticle(initialPosition, initialCost = 0):
    particle = {
        'position': initialPosition,
        'positionFitness': initialCost,
        'velocity': [],
        'bestPosition': initialPosition,
        'bestFitness': initialCost,
        'bestGlobalFitness': 0,
        'bestGlobalPosition': []
    }
    return particle

def initializeSwarm(populationSize):
    swarm = []
    for i in range(populationSize):
        particlePosition = initSingleParticle()
        particle = createParticle(particlePosition)
        swarm.append(particle)
    # print(swarm)
    return swarm

def initSingleLayerFC():
    randFc = random.random()
    # print('randFc', randFc)
    layer = None
    if randFc >= 0.5:
        valueFC = random.randint(3, 10)
        layer = {
            'layerType': 'FC',
            'layerNumber': valueFC
        }
    return layer

def initSingleLayerDropout():
    randDropout = random.random()
    # print('randDropout', randDropout)
    layer = None
    if randDropout >= 0.5:
        valueDropout = random.uniform(0, 0.6)
        layer = {
            'layerType': 'Dropout',
            'layerNumber': valueDropout
        }
    return layer

def initSingleLayerLR():
    valueLR = random.randint(1, 6)
    layer = {
        'layerType': 'LR',
        'layerNumber': valueLR
    }
    return layer

def initFCLayers():
    #Init all convolutional and dropout layers
    layers = []
    layersIndex = 0
    for i in range(5):
        initialSeed = random.uniform(0, 5000)
        random.seed(initialSeed)
        
        # print('\n\ni', i, 'layersIndex', layersIndex)
        fc = initSingleLayerFC()
        dropout = initSingleLayerDropout()
        
        # print('layers', layers)
        # print('fc', fc, 'dropout', dropout)

        if fc != None:
            layers.append(fc)
            layersIndex = layersIndex + 1

        if dropout != None and layersIndex > 0 and layers[layersIndex-1]['layerType'] == 'FC':
            layersIndex = layersIndex + 1
            layers.append(dropout)

    #this layer will be placed only when transforming the pso into cnn
    #in order to simplify the position update
    # layer = {
    #     'layerType': 'FC',
    #     'layerNumber': 1
    # }
    # layers.append(layer)

    # print('layers', layers)
    return layers

def initSingleParticle():
    initialSeed = random.uniform(0, 5000)
    random.seed(initialSeed)

    particle = []

    #Init first position with LR
    lrLayer = initSingleLayerLR()
    # print('lrLayer', lrLayer)
    particle.append(lrLayer)

    layers = initFCLayers() 

    particle.extend(layers)
    print('particle', particle)
    return particle
