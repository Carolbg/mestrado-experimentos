class PSOCacheClass:
    def __init__(self):
        self.cacheStore = {}

    def verifyEntry(self, particlePosition):    
        # print('verifyEntry = cacheStore', self.cacheStore)
        individuoAsStr = repr(tuple(particlePosition))
        print('verifyEntry', individuoAsStr)
        
        if individuoAsStr in self.cacheStore.keys():
            print('achei ',individuoAsStr,' no cache')
            return self.cacheStore[individuoAsStr]
        return None
    
    def getCache(self):
        return self.cacheStore

    def addNewEntry(self, particlePosition, fitnessValue):
        individuoAsStr = repr(tuple(particlePosition))
        # print('key ', individuoAsStr, '\n')
        self.cacheStore[individuoAsStr] = fitnessValue
        
    def savePopulationToCache(self, swarm):
        print('fitness = ', self.cacheStore)
        [self.addNewEntry(swarm[i]['position'], swarm[i]['positionFitness']) for i in range(len(swarm))]
        print('savePopulationToCache = cacheStore', self.cacheStore)
