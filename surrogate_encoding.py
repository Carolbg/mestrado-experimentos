import copy

def encodeAGIndividual(agIndividual):
    encodedIndividual = []

    for i in range(len(agIndividual)):
        cromossome = agIndividual[i]
        # print('cromossome = ', cromossome)
        if i == 0: #LR cromossome
            rfPart = [1, 1, cromossome[0]]
        else:
            if i%2 != 0:  # FC cromossome
                rfPart = [2, cromossome[0], cromossome[1]]
            else: #Dropout cromossome
                rfPart = [3, cromossome[0], cromossome[1]]
        encodedIndividual.append(rfPart)

    encodedIndividual.append([2, 1, 2]) #adding the final layer
    # print('encodedIndividual', encodedIndividual)    

    # making the rule of the size*3 as shown in the article
    parsedEncodedIndividual = copy.deepcopy(encodedIndividual)
    size = len(parsedEncodedIndividual)
    for i in range(size, size*3):
        parsedEncodedIndividual.append([0, 0, 0])
    
    # print('parsedEncodedIndividual', parsedEncodedIndividual)
    
    return encodedIndividual


# def encodeAGIndividual(agIndividual):
#     encodedIndividual = []
#     #The encoded individual will be [type, value], where type is 1 for LR
#     # 2 for FC and 3 for D

#     size = len(agIndividual)
#     i = 0
#     while i < size:
#         cromossome = agIndividual[i]
#         if i == 0: #LR cromossome
#             rfPart = [1, cromossome[0]]
#             i+=1
#         else:
#             #when the isPresent is false, for the sake of simplicity we are removing
#             #if the isPresent is false from an FC layer, then the following dropout
#             #layer is also ignored.
#             if cromossome[0] == 0: 
#                 if i%2 != 0: #FC layer, therefore remove the following dropout layer
#                     i+=2
#                 else: #dropout layer
#                     i+=1
#                 continue

#             if i%2 != 0:  # FC cromossome
#                 rfPart = [2, cromossome[1]]
#                 i+=1
#             else: #Dropout cromossome
#                 rfPart = [3, cromossome[1]]
#                 i+=1
#         encodedIndividual.append(rfPart)
    
#     encodedIndividual.append([2, 2]) #adding the final layer
#     # print('encodedIndividual', encodedIndividual)    

#     # making the rule of the size*2 as shown in the article
#     parsedEncodedIndividual = copy.deepcopy(encodedIndividual)
#     size = len(parsedEncodedIndividual)
#     for i in range(size, size*2):
#         parsedEncodedIndividual.append([0,0])
    
#     print('parsedEncodedIndividual', parsedEncodedIndividual)
    
#     return parsedEncodedIndividual
