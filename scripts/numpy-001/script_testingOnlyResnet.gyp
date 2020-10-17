#!/usr/bin/env python3

import sys
sys.path.append('../../')

from mainVGG import *
from mainResnet import *
from mainDensenet import *

# resultsPlotName = 'matlabPreprocessing_1camada_10epocas_lr'
resultsPlotName = 'matlabPreprocessing_1camada'
experimentType = 1
dataAugmentation = False
typeLR = 2

print('Config: ', resultsPlotName)
print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

try:
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR)
except Exception as e:
    print('Error in resnet', e)
