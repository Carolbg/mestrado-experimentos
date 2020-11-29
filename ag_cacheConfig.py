def initCache():
    global cacheStore
    cacheStore = {}

def verifyEntry(individuo):
    #retirar o que tem 0 que acaba nao aplicando para compor a chave
    keyArray = []
    for i in range(len(individuo)):
        if i <2:
            keyArray.append(individuo[i])
        elif individuo[i][0] == 1:
            keyArray.append(individuo[i])
    # print('keyArray', keyArray)
    
    individuoAsStr = repr(tuple(individuo))
    # print('verifyEntry = cacheStore', cacheStore)

    if individuoAsStr in cacheStore.keys():
        return cacheStore[individuoAsStr]
    return None

def addNewEntry(individuo, fitnessValue):
    individuoAsStr = repr(tuple(individuo))

    global cacheStore
    cacheStore[individuoAsStr] = fitnessValue
    print('addNewEntry = cacheStore', cacheStore)