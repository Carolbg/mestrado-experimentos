#!/usr/bin/env python3

import sys
sys.path.append('../../')

from mainVGG import *
from mainResnet import *
from mainDensenet import *

resultsPlotName = 'matlabPreprocessing_1camada_10epocas_lr'
# resultsPlotName = 'matlabPreprocessing_1camada'
experimentType = 1
dataAugmentation = False
typeLR = 1

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
