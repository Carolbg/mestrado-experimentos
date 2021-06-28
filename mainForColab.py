from mainVGG import *
from mainResnet import *
from mainDensenet import *

import gc
import torch

# resultsPlotName = 'minMax_1camada'
# experimentType = 1
# typeLR = 2
# dataAugmentation = False
def main(resultsPlotName, experimentType, typeLR=1, isNumpy=False, dataAugmentation=False):

    print('Config: ', resultsPlotName)
    print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

    try:
        mainVGG(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy)
        
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print('Error in vgg', e)

    try:
        mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy)
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print('Error in resnet', e)

    try: 
        mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy)
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print('Error in densenet', e)


def runVGG(resultsPlotName, experimentType, typeLR=1, isNumpy=False,dataAugmentation=False):

    print('Config: ', resultsPlotName)
    print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

    try:
        model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainVGG(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print('Error in vgg', e)
    

def runResnet(resultsPlotName, experimentType, typeLR=1,isNumpy=False, dataAugmentation=False):

    print('Config: ', resultsPlotName)
    print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

    try:
        model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print('Error in resnet', e)

def runDensenet(resultsPlotName, experimentType, typeLR=1, isNumpy=False,dataAugmentation=False):

    print('Config: ', resultsPlotName)
    print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

    try: 
        model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print('Error in densenet', e)
