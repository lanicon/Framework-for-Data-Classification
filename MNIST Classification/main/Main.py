'''
@author: Paul Fung
'''

import sys

from datastore.DataLoader import MatLoader
from datastore.DataStore import DataStore
import numpy as np
import scipy.io as sio
import optparse as op
from visualization.DisplayData import displayData

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

#constants
CONST_NN_WEIGHTS_PATH = "../input/nnWeights.mat"
CONST_LR_WEIGHTS_PATH = "../input/lrWeights.mat"
CONST_DATA_PATH = "../input/data.mat"

CONST_NN_WEIGHT1_NAME = "Theta1"
CONST_NN_WEIGHT2_NAME = "Theta2"

CONST_LR_WEIGHT_NAME = "all_theta"

CONST_DATA_FEATURES_NAME = "X"
CONST_DATA_LABEL_NAME = "y"

CONST_NUM_PREDICTS = 10

TRAIN_DATA_PROP = 0.6
CV_DATA_PROP = 0.2
TEST_DATA_PROP = 1 - TRAIN_DATA_PROP - CV_DATA_PROP

def main(args):
    
    #parse program arguments
    parser = op.OptionParser()
    parser.set_defaults(LR=False,NN=False,Train=False)
    parser.add_option('--LR', action='store_true', dest='LR')
    parser.add_option('--NN', action='store_true', dest='NN')
    parser.add_option('--Train', action='store_true', dest='Train')
    (options, args) = parser.parse_args()
    
    if (options.LR and options.NN):
        print("Only one classifier is allowed")
        exit()
        
    if ((not options.LR) and (not options.NN)):
        print("Use --LR or --NN to choose the classifier")
        exit()
    
    if options.LR:
        print("LR flag")

    if options.NN:
        options.LR = False
        print("NN flag")
        
    if options.Train:
        print("Train flag")
        
    #loads features and labels
    matLoader = MatLoader()
    matContent = matLoader.loadmat(filename=CONST_DATA_PATH)
    Data_X = matContent['X']
    Data_y = matContent['y'].flatten()-1
    
    #column join and shuffle
    Data_Xy = np.c_[Data_X, Data_y]
    np.random.shuffle(Data_Xy)
    Data_X = Data_Xy[:,0:Data_X.shape[1]]
    Data_y = Data_Xy[:,Data_X.shape[1]:].flatten().astype(int)
    
    #segregate into training, cross validation and test set
    numRecords = Data_X.shape[0]
    DataStore.training_set_X = Data_X[0:int(numRecords*TRAIN_DATA_PROP),:]
    DataStore.training_set_y = Data_y[0:int(numRecords*TRAIN_DATA_PROP)]
    DataStore.CV_set_X = Data_X[int(numRecords*TRAIN_DATA_PROP):int(numRecords*(TRAIN_DATA_PROP+CV_DATA_PROP)),:]
    DataStore.CV_set_y = Data_y[int(numRecords*TRAIN_DATA_PROP):int(numRecords*(TRAIN_DATA_PROP+CV_DATA_PROP))]
    DataStore.test_set_X = Data_X[int(numRecords*(TRAIN_DATA_PROP+CV_DATA_PROP)):,:]
    DataStore.test_set_y = Data_y[int(numRecords*(TRAIN_DATA_PROP+CV_DATA_PROP)):]
    numOfLabels = len(np.unique(DataStore.training_set_y))

    #regularization and optimization parameters
    lambdaVals = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    maxIter = 50
    
    classifier = None
    
    if options.NN:
        print("Using Neural Network Classifier...")
        
        #import
        from classifier.NNClassifier import NNClassifier
        
        #to trainAndTuneLambda for theta values
        if options.Train:
            classifier = NNClassifier()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                retVal = classifier.trainAndTuneLambda(X_train=DataStore.training_set_X, y_train=DataStore.training_set_y, 
                                              X_cv=DataStore.CV_set_X, y_cv=DataStore.CV_set_y,
                                              numOfLabels = numOfLabels, lambdaVals=lambdaVals, maxIter = maxIter)
            print(retVal)    
        
        #to load theta values
        else:
            matContent = sio.loadmat(CONST_NN_WEIGHTS_PATH)
            classifier = NNClassifier(matContent[CONST_NN_WEIGHT1_NAME], matContent[CONST_NN_WEIGHT2_NAME])
 
    elif options.LR:
        print("Using Logistic Regression Classifier...")
        
        #import
        from classifier.LRClassifier import LRClassifier
        
        #to trainAndTuneLambda theta values
        if options.Train:
            classifier = LRClassifier()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                retVal = classifier.trainAndTuneLambda(X_train=DataStore.training_set_X, y_train=DataStore.training_set_y, 
                                              X_cv=DataStore.CV_set_X, y_cv=DataStore.CV_set_y,
                                              numOfLabels = numOfLabels, lambdaVals=lambdaVals, maxIter = maxIter)
            print(retVal)   
         
        #to load theta values   
        else:
            matContent = sio.loadmat(CONST_LR_WEIGHTS_PATH)        
            classifier = LRClassifier(matContent[CONST_LR_WEIGHT_NAME])
        
    else:
        print("Invalid argument, exiting")
        exit()
    
    #returns all predictions of training set
    predictions = classifier.predict(DataStore.test_set_X)
    
    #compares predictions with labels
    accuracy = np.mean(predictions == DataStore.test_set_y.conjugate().T) * 100
    
    if options.Train:
        print("With training set size %d, max iteration %d" % (numRecords*TRAIN_DATA_PROP,maxIter))
        
    print("%s accuracy for MNIST test set: %f percent" % (classifier.name,accuracy))
    
    #classifies the digit in an image using the specified classifier
    print("\nRunning the classifier %d times to predict the digits in the random images:" % CONST_NUM_PREDICTS)
    m = len(DataStore.test_set_X)
    for i in range(CONST_NUM_PREDICTS):
        idx = np.random.randint(m)
        sel = np.matrix(DataStore.test_set_X[idx,:])
        prediction = classifier.predict(sel)
        
        print("%d: %s prediction: %d (close display window to continue)" % (i+1, classifier.name,np.mod(prediction+1, 10)))
        displayData(sel)
        
    print("End of Program")
    

if __name__ == "__main__":
    main(sys.argv)