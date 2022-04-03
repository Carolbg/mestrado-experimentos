from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor # for building the model
import numpy as np 

# https://medium.com/@theclickreader/random-forest-regression-explained-with-implementation-in-python-3dad88caf165

def createRandomForest(numberTree=10):
    model = RandomForestRegressor(n_estimators = numberTree, random_state = 0)
    return model

def trainModel(model, xTrain, yTrain):
    print('no train model xTrain = ', xTrain)
    # Fitting the Random Forest Regression model to the data
    return model.fit(xTrain, yTrain)

def testModel(model, xTest):
    print('no testModel xTest = ', xTest)
    # Predicting the target values of the test set
    yPred = model.predict(xTest)
    return yPred

def calcRMSE(yTest, yPred):
    # RMSE (Root Mean Square Error)
    rmse = float(format(np.sqrt(mean_squared_error(yTest, yPred)),'.3f'))
    print("\nRMSE:\n",rmse)