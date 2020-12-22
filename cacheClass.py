from ag_cacheConfig import parseIndividuoToKey

class CacheClass:
    def __init__(self):
        self.cacheStore = {}

    def verifyEntry(self, individuo):    
        print('verifyEntry = cacheStore', self.cacheStore)
        individuoAsStr = parseIndividuoToKey(individuo)
        print('verifyEntry', individuoAsStr)
        
        if individuoAsStr in self.cacheStore.keys():
            print('achei ',individuoAsStr,' no cache')
            return self.cacheStore[individuoAsStr]
        return None

    def addNewEntry(self, individuo, fitnessValue):
        individuoAsStr = parseIndividuoToKey(individuo)
        print('key ', individuoAsStr, '\n')
        self.cacheStore[individuoAsStr] = fitnessValue
        

    def savePopulationToCache(self, population, fitness):
        print('fitness = ', self.cacheStore)
        [self.addNewEntry(population[i], fitness[i]) for i in range(len(population))]
        print('savePopulationToCache = cacheStore', self.cacheStore)
