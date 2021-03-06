def verifyNetworkLayers(individuo):
    # print('individuo', individuo, flush=True)
    
    numberNeurons = []

    for i in range(1, len(individuo)):
        #Considerando apenas as camadas densamente conectadas
        if i %2 != 0:
            # print('camada',i, 'eh densamente conectada', flush=True)
            if individuo[i][0] == 1:
                # print('camada ', i, ' esta presente', flush=True)
                numberNeurons.append(individuo[i][1])
            else:
                # print('camada ', i, ' NAO esta presente', flush=True)

    # print('numberNeurons', numberNeurons, flush=True)
    isReducingLayerSize = False
    
    if(all(numberNeurons[i] >= numberNeurons[i + 1] for i in range(len(numberNeurons)-1))): 
        isReducingLayerSize = True
    # print('isReducingLayerSize', isReducingLayerSize, flush=True)
    
    return isReducingLayerSize