def initCache():
    global cacheStore
    cacheStore = {}

def parseIndividuoToKey(individuo):
    #retirar o que tem 0 que acaba nao aplicando para compor a chave
    keyArray = []
    for i in range(len(individuo)):
        if i <2:
            keyArray.append(individuo[i])
        elif individuo[i][0] == 1:
            keyArray.append(individuo[i])
    
    individuoAsStr = repr(tuple(keyArray))
    # print('keyArray', keyArray)
    return individuoAsStr

def verifyEntry(individuo):
    
    # print('verifyEntry = cacheStore', cacheStore)
    individuoAsStr = parseIndividuoToKey(individuo)

    if individuoAsStr in cacheStore.keys():
        # print('achei ',individuoAsStr,' no cache')
        return cacheStore[individuoAsStr]
    return None

def addNewEntry(individuo, fitnessValue):
    individuoAsStr = parseIndividuoToKey(individuo)

    global cacheStore
    cacheStore[individuoAsStr] = fitnessValue
    # print('addNewEntry = cacheStore', cacheStore)
    

def savePopulationToCache(population, fitness):
    [addNewEntry(population[i], fitness[i]) for i in range(len(population))]
