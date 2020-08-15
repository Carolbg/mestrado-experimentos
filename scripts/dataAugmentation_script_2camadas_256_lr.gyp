#!/usr/bin/env python3

import sys
sys.path.append('../')

from mainVGG import *
from mainResnet import *
from mainDensenet import *

# resultsPlotName = 'dataAugmentation_matlabPreprocessing_2camadas_256_lr'
resultsPlotName = 'dataAugmentation_matlabPreprocessing_2camadas_256'
experimentType = 2
dataAugmentation = True
typeLR = 2

print('Config: ', resultsPlotName)
print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

try:
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainVGG(resultsPlotName, experimentType, dataAugmentation, typeLR)
except Exception as e:
    print('Error in vgg', e)

try:
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR)
except Exception as e:
    print('Error in resnet', e)

try: 
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR)
except Exception as e:
    print('Error in densenet', e)