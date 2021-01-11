from mainVGG import *
from mainResnet import *
from mainDensenet import *

# resultsPlotName = 'minMax_1camada'
# experimentType = 1
# typeLR = 2
# dataAugmentation = False
def main(resultsPlotName, experimentType, typeLR=1, dataAugmentation=False):

    print('Config: ', resultsPlotName)
    print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

    try:
        model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainVGG(resultsPlotName, experimentType, dataAugmentation, typeLR, False)
    except Exception as e:
        print('Error in vgg', e)

    try:
        model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR, False)
    except Exception as e:
        print('Error in resnet', e)

    try: 
        model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR, False)
    except Exception as e:
        print('Error in densenet', e)
