'''
@author: Paul Fung
'''

import unittest

from scipy import io as sio

from classifier.NNClassifier import NNClassifier
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
        matContent = sio.loadmat(CONST_NN_WEIGHTS_PATH)        
        nn = NNClassifier(matContent[CONST_NN_WEIGHT1_NAME], matContent[CONST_NN_WEIGHT2_NAME])
        #print(nn.weights1.shape)
        matContent = sio.loadmat(CONST_DATA_PATH)
        features = matContent[CONST_DATA_FEATURES_NAME]
        y = matContent[CONST_DATA_LABEL_NAME]      
        y = y[:,0] - 1
        
        #predict
        predictions = nn.predict(features)
        accuracy = np.mean(predictions == y.conjugate().T)
        print("NN Accuracy: ", accuracy)
        
    def testPredictWithTrg(self):
        #load training set
        matContent = sio.loadmat(CONST_DATA_PATH)
        features = matContent[CONST_DATA_FEATURES_NAME]
        y = matContent[CONST_DATA_LABEL_NAME]      
        y = y[:,0] - 1
        
        #define nn and training parameters
        inputLayerSize = features.shape[1]
        hiddenLayerSize = 25
        numLabels = len(np.unique(y))
        lambdaVal = 3
        maxIter = 50

        #training
        nn = NNClassifier()
        nn.train(inputLayerSize = inputLayerSize, 
                 hiddenLayerSize = hiddenLayerSize, 
                 numLabels = numLabels, 
                 X = features, y = y, 
                 lambdaVal = lambdaVal, 
                 maxIter = maxIter)
        
        #predict
        predictions = nn.predict(features)
        accuracy = np.mean(predictions == y.conjugate().T)
        print("NN Accuracy (trg): ", accuracy)
         
    def testFunctions(self):
        nn = NNClassifier()
         
        #test random init weight
        randomW = nn.randomInitWeights(4, 6)
        print("RandomInitializedWeights:")
        print(randomW)
         
        #test debug init weight
        debugW = nn.debugInitWeights(4, 6)
        print("DebugInitializedWeights:")
        print(debugW)
         
        #test sigoid computeGradient
        print("sigmoidGrad: ", nn.sigmoidGradient(0))
         
        #test cost function
        matContent = sio.loadmat(CONST_DATA_PATH)
        X = matContent[CONST_DATA_FEATURES_NAME]
        y = matContent[CONST_DATA_LABEL_NAME]
        y = y[:,0] - 1
         
        matContent = sio.loadmat(CONST_NN_WEIGHTS_PATH)
        nn = NNClassifier()
         
        inputLayerSize = X.shape[1]
        hiddenLayerSize = 25
        numLabels = len(np.unique(y))
        theta1 = matContent[CONST_NN_WEIGHT1_NAME].flatten()
        theta2 = matContent[CONST_NN_WEIGHT2_NAME].flatten()
        lambdaVal = 3
        nnParams = np.append(theta1, theta2)
        cost = nn.computeCost(inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, nnParams)
        print("cost: ", cost)
         
        #test training
        maxIter = 50
        nn.train(inputLayerSize = inputLayerSize, 
                 hiddenLayerSize = hiddenLayerSize, 
                 numLabels = numLabels, 
                 X = X, y = y, 
                 lambdaVal = lambdaVal, 
                 maxIter = maxIter)
         
    def testGrad(self):
        #creates dummy nn
        inputLayerSize = 3
        hiddenLayerSize = 5
        numLabels = 3
        nn = NNClassifier()
        #arbitrary values for theta1 and theta2
        theta1 = nn.debugInitWeights(inputLayerSize, hiddenLayerSize)
        theta2 = nn.debugInitWeights(hiddenLayerSize, numLabels)
        m = 5
        
        #arbitrary values for X and y
        X = nn.debugInitWeights(inputLayerSize - 1, m)
        y = list(i % numLabels for i in list(range(0,m)))
        lambdaVal = 3
        nnParams = np.append(theta1.flatten(), theta2.flatten())
         
        #compute and compare computeGradient values
        shortCostFunction = lambda nnParams : nn.computeCost(inputLayerSize, hiddenLayerSize, 
                                                              numLabels, X, y, lambdaVal, nnParams)
        grad = nn.computeGradient(inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, nnParams)
        numGrad = nn.computeNumericalGradient(shortCostFunction, nnParams)
        print("grad: ", grad)
        print("numGrad: ", numGrad)
        diff = np.linalg.norm(grad-numGrad)/np.linalg.norm(grad+numGrad)
        print("diff: ", diff)

