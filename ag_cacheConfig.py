def initCache():
    global cacheStore
    cacheStore = {}

def parseIndividuoToKey(individuo):
    #retirar o que tem 0 que acaba nao aplicando para compor a chave
    keyArray = []

    tam = len(individuo)
    i=0
    while i < tam:
        # print('i', i, 'individuo2[i]', individuo2[i])
        if i < 2:
            keyArray.append(individuo[i])
            # print('if 1')
            i = i+1
        else:
            if i%2 == 0 and individuo[i][0] == 0:
                i = i+2
                # print('if 2')
            elif individuo[i][0] == 0:
                i = i+1
            else:
                keyArray.append(individuo[i])
                i = i+1
                # print('else')
        # input('')
    
    individuoAsStr = repr(tuple(keyArray))
    # print('keyArray', keyArray)
    return individuoAsStr

def verifyEntry(individuo):
    
    # print('verifyEntry = cacheStore', cacheStore)
    individuoAsStr = parseIndividuoToKey(individuo)
    print('verifyEntry', individuoAsStr)
    
    if individuoAsStr in cacheStore.keys():
        # print('achei ',individuoAsStr,' no cache')
        return cacheStore[individuoAsStr]
    return None

def addNewEntry(individuo, fitnessValue):
    individuoAsStr = parseIndividuoToKey(individuo)
    print('\nindividuo', individuo, ' key ', individuoAsStr, '\n')
    global cacheStore
    cacheStore[individuoAsStr] = fitnessValue
    # print('addNewEntry = cacheStore', cacheStore)
    

def savePopulationToCache(population, fitness):
    [addNewEntry(population[i], fitness[i]) for i in range(len(population))]
