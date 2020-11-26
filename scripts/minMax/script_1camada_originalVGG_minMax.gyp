#!/usr/bin/env python3

import sys
sys.path.append('../../')

from mainVGG import *
from mainResnet import *
from mainDensenet import *

resultsPlotName = 'originalVGG_minMax_1camada'
experimentType = 1
dataAugmentation = False
typeLR = 2

print('Config: ', resultsPlotName)
print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

try:
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainVGG(resultsPlotName, experimentType, dataAugmentation, typeLR, False, True)
except Exception as e:
    print('Error in vgg', e)
