'''
@author: Paul Fung
'''

import unittest

from scipy import io as sio

from classifier.LRClassifier import LRClassifier
import numpy as np

CONST_NN_WEIGHTS_PATH = "../../input/nnWeights.mat"
CONST_LR_WEIGHTS_PATH = "../../input/lrWeights.mat"
CONST_DATA_PATH = "../../input/data.mat"

CONST_NN_WEIGHT1_NAME = "Theta1"
CONST_NN_WEIGHT2_NAME = "Theta2"

CONST_LR_WEIGHT_NAME = "all_theta"

CONST_DATA_FEATURES_NAME = "X"
CONST_DATA_LABEL_NAME = "y"

CONST_NN_OPTION = "NN"
CONST_LR_OPTION = "LR"

class Test(unittest.TestCase):

    def testPredict(self):
        #load data
        matContent = sio.loadmat(CONST_LR_WEIGHTS_PATH)
        lr = LRClassifier(matContent[CONST_LR_WEIGHT_NAME])
        #print(lr.weights.shape)
        
        matContent = sio.loadmat(CONST_DATA_PATH)
        features = matContent[CONST_DATA_FEATURES_NAME]
        y = matContent[CONST_DATA_LABEL_NAME]
        y = y[:,0] - 1
        
        #predict
        predictions = lr.predict(features)
        #print(predictions)
        #print(y+1)
        
        accuracy = np.mean(predictions == y.conjugate().T)
        print("LR Accuracy: ", accuracy)
        
    def testPredictWithTrg(self):        
        lr = LRClassifier()
        
        #load data
        matContent = sio.loadmat(CONST_DATA_PATH)
        features = matContent[CONST_DATA_FEATURES_NAME]
        y = matContent[CONST_DATA_LABEL_NAME]
        y = y[:,0] - 1
        lambdaVal = 0
        maxIter = 50
        numOfLabels = len(np.unique(y))
        
        #train
        lr.train(X=features, y=y, numOfLabels = numOfLabels, lambdaVal=lambdaVal, maxIter = maxIter)
          
        predictions = lr.predict(features)
        #print(predictions)
        #print(y+1)
          
        accuracy = np.mean(predictions == y.conjugate().T)
        print("LR Accuracy (trg): ", accuracy)
        
    def testGrad(self):
        thetaSize = 10
        numOfFeatures = 10
        m = 5
        lr = LRClassifier()
        #arbitrary values for theta
        theta = lr.debugInitWeights(thetaSize)
        
        #arbitrary values for X and y
        X = np.reshape(lr.debugInitWeights(numOfFeatures*m), (m,numOfFeatures))
        y = np.reshape(list(i % 2 for i in list(range(0,m))), (1,-1))
        lambdaVal = 3
        
        #compute and compare computeGradient values
        shortCostFunction = lambda theta : lr.computeCost(X, y, lambdaVal, theta)
        grad = lr.computeGradient(X, y, lambdaVal, theta)
        numGrad = lr.computeNumericalGradient(shortCostFunction, theta)
        
        print("grad: ", grad)
        print("numGrad: ", numGrad)
        diff = np.linalg.norm(grad-numGrad)/np.linalg.norm(grad+numGrad)
        print("diff: ", diff)
