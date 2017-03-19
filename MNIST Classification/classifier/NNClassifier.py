'''
@author: Paul Fung
'''

from scipy.optimize.optimize import fmin_cg

from classifier.BaseClassifier import BaseClassifier
import numpy as np


class NNClassifier(BaseClassifier):
    '''
    Neural Network Classifier
    '''

    def __init__(self, *params):
        """
        Initialization of the Neural Network classifier
        
        Parameters
        ----------
        *params : list
        item 1 and item 2 (optional) - contains the initial values of theta1 and theta2. Either none or both values should be provided
        """
        super().__init__(*params)
        self.name = "Neural Network Classifier"
        self.theta1 = None
        self.theta2 = None
        
        if len(params)==0:
            return
        elif len(params)==2:
            self.theta1 = params[0]
            self.theta2 = params[1]
        else:
            print("Incorrect argument number, exiting")
            exit()
    
    
    def predict(self, features):
        """
        Prediction of label using the input features
        
        Parameters
        ----------
        features : ndarray (2D)
            Input variable for predicting its class
            
        Returns:
        --------
        p : ndarray (1D)
            The predicted classes
        """
        m = features.shape[0]
        
        #prepend a '1' column
        X = np.c_[np.ones(m), features]
        
        #calculations for hidden layer
        a2 = self.sigmoid(X.dot(self.theta1.conjugate().T))
        m = a2.shape[0]
        a2 = np.c_[np.ones(m), a2]
        
        #calculations for output layer
        a3 = self.sigmoid(a2.dot(self.theta2.conjugate().T))
        
        #extracts the index of highest value for every record
        p = a3.argmax(axis=1)
        return p
    
        
    def train(self, update=True, **params):
        """
        Training of model to generate the theta value for the this classifier
        
        Parameters
        ----------
        update : boolean
            when set to true, the classifier's theta value is updated                
        **params : list
            inputLayerSize : int
                Number of input features
            hiddenLayerSize : int
                Number of nodes in the hidden layer
            numLabels : int
                Number of unique labels (i.e. classes)
            X : ndarray (2D)
                Contains the training set, with each row as one record
            y : ndarray (1D)
                Contains the corresponding label for each row in X
            lambdaVal : float
                Regularization parameter
            maxIter : int
                Maximum number of iterations that the optimization algorithm will run for each label
                
        Returns:
        --------
        xopt : ndarray (1D)
            optimized theta value
        cost : float
            cost associated to xopt                   
        """    
        inputLayerSize = params["inputLayerSize"]
        hiddenLayerSize = params["hiddenLayerSize"]
        numLabels = params["numLabels"]
        X = params["X"]
        y = params["y"]
        lambdaVal = params["lambdaVal"]
        maxIter = params["maxIter"]

        theta1 = self.randomInitWeights(inputLayerSize, hiddenLayerSize)
        theta2 = self.randomInitWeights(hiddenLayerSize, numLabels)
        nnParams = np.append(theta1, theta2)

        shortCostFunction = lambda nnParams : self.computeCost(inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, nnParams)
        shortGradFunction = lambda nnParams : self.computeGradient(inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, nnParams)
        
        retVal = fmin_cg(shortCostFunction, x0=nnParams, fprime=shortGradFunction, maxiter=maxIter, full_output=True)
        nnParams = retVal[0]
        
        if update:
            self.theta1 = np.reshape(nnParams[0:hiddenLayerSize*(inputLayerSize+1)], (hiddenLayerSize, inputLayerSize+1))
            self.theta2 = np.reshape(nnParams[hiddenLayerSize*(inputLayerSize+1):], (numLabels, hiddenLayerSize+1))
        
        retVal = (retVal[0], retVal[1])
        return retVal
    
    
    def trainAndTuneLambda(self, update=True, **params):
        """
        Training of model and tuning of regularization term, for generating the theta and lambda value for the this classifier
        
        Parameters
        ----------
        update : boolean
            when set to true, the classifier's theta value is updated        
        **params : list
            X_train : ndarray (2D)
                Contains the training set, with each row as one record
            y_train : ndarray (1D)
                Contains the corresponding label for each row in X_train
            X_cv : ndarray (2D)
                Contains the cross validation set, with each row as one record
            y_cv : ndarray (1D)
                Contains the corresponding label for each row in X_cv          
            lambdaVals : ndarray (1D)
                Regularization parameters to test
            maxIter : int
                Maximum number of iterations that the optimization algorithm will run for each label
            numOfLabels : int
                Number of unique labels

        Returns:
        --------
        lambdaValCost : ndarray (2D)
            costs associated with each item in lambdaValCost
        """                
        X_train = params["X_train"]
        y_train = params["y_train"]
        X_cv = params["X_cv"]
        y_cv = params["y_cv"]
        lambdaVals = params["lambdaVals"]
        maxIter = params["maxIter"]
        numLabels = params["numOfLabels"]
        
        #define nn, regularization and optimization parameters
        inputLayerSize = X_train.shape[1]
        hiddenLayerSize = 25
        #numLabels = len(np.unique(y_train))
        
        lambdaValCost = np.zeros((len(lambdaVals), 2))
        minCost = float("inf")
        minCostLambdaVal = 0
        minCostTheta = None
        
        for i in range(0,len(lambdaVals)):
            print("lambdaVal: ", lambdaVals[i])
            retVal = self.train(update=False, X=X_train, y=y_train, lambdaVal=lambdaVals[i], 
                                maxIter=maxIter, numLabels=numLabels, inputLayerSize = inputLayerSize, 
                                hiddenLayerSize = hiddenLayerSize)
            lambdaValCost[i,0] = lambdaVals[i]
            theta_train = retVal[0]
            
            lambdaValCost[i,1] = self.computeCost(inputLayerSize, hiddenLayerSize, numLabels, X_cv, y_cv, lambdaVals[i], theta_train)
            print("currCost: ", lambdaValCost[i,1])
            
            if(lambdaValCost[i,1] < minCost):
                minCost = lambdaValCost[i,1]
                minCostTheta = theta_train
                minCostLambdaVal = lambdaVals[i]
                
        print("minCostLambdaVal: ", minCostLambdaVal)
        
        if update:
            self.theta1 = np.reshape(minCostTheta[0:hiddenLayerSize*(inputLayerSize+1)], (hiddenLayerSize, inputLayerSize+1))
            self.theta2 = np.reshape(minCostTheta[hiddenLayerSize*(inputLayerSize+1):], (numLabels, hiddenLayerSize+1))
        
        return lambdaValCost    
    
    
    def randomInitWeights(self, L_in, L_out):
        """
        Random initiation of weights
        
        Parameters
        ----------
        L_in : int
            Input layer size
        L_out : int
            Output layer size
            
        Returns
        -------
        w : ndarray (2D)
            theta of size (L_out, 1+L_in) with initialized values (weights)
        """      
        epsilon_init = 0.12
        w = 2 * np.random.random((L_out, 1+L_in)) * epsilon_init - epsilon_init
        return w
    
    
    def debugInitWeights(self, L_in, L_out):
        """
        Fixed initiation of weights
        
        Parameters
        ----------
        L_in : int
            Input layer size
        L_out : int
            Output layer size
            
        Returns
        -------
        w : ndarray (2D)
            theta of size (L_out, 1+L_in) with initialized values (weights)
        """      
        w = np.sin(np.reshape(list(range(1, (L_out)*(1+L_in)+1)), (L_out, 1+L_in)))/10
        return w
    
    
    def computeCost(self, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, nnParams):
        """
        Computation of cost
        
        Parameters
        ----------
        inputLayerSize : int
            Number of input features
        hiddenLayerSize : int
            Number of nodes in the hidden layer
        numLabels : int
            Number of unique labels (i.e. classes)
        X : ndarray (2D)
            Contains the training set, with each row as one record
        y : ndarray (1D)
            Contains the corresponding label for each row in X
        lambdaVal : float
            Regularization parameter
        nnParams : ndarray (1D)
            values for theta1 and theta2
            
        Returns
        -------
        J : float
            cost value
        """   
        #1 input layer, 1 hidden layer
        theta1 = np.reshape(nnParams[0:hiddenLayerSize*(inputLayerSize+1)], (hiddenLayerSize, inputLayerSize+1))
        theta2 = np.reshape(nnParams[hiddenLayerSize*(inputLayerSize+1):], (numLabels, hiddenLayerSize+1))
        
        m = X.shape[0]
        
        #cost function
        a1 = np.c_[np.ones(m), X]
        z2 = a1.dot(theta1.conjugate().T)
        a2 = np.c_[np.ones(m), self.sigmoid(z2)]
        z3 = a2.dot(theta2.conjugate().T)
        a3 = self.sigmoid(z3)
        
        #expands y
        y = np.eye(numLabels)[:,y].conjugate().T  
        
        J = 1/m * sum(sum(np.multiply(-y, np.log(a3)) - np.multiply(1-y, np.log(1-a3))))
        
        #regularization
        J += lambdaVal/(2*m) * (sum(sum(np.power(theta1[:,1-theta1.shape[1]:],2))) + sum(sum(np.power(theta2[:,1-theta2.shape[1]:],2))))
        
        return J
    
    
    def computeGradient(self, inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, nnParams):
        """
        Computation of gradient
        
        Parameters
        ----------
        inputLayerSize : int
            Number of input features
        hiddenLayerSize : int
            Number of nodes in the hidden layer
        numLabels : int
            Number of unique labels (i.e. classes)
        X : ndarray (2D)
            Contains the training set, with each row as one record
        y : ndarray (1D)
            Contains the corresponding label for each row in X
        lambdaVal : float
            Regularization parameter
        nnParams : ndarray (1D)
            values for theta1 and theta2
            
        Returns
        -------
        grad : ndarray (1D)
            value of theta1 and theta2 joined in a 1D ndarray
        """   
        #1 input layer, 1 hidden layer
        theta1 = np.reshape(nnParams[0:hiddenLayerSize*(inputLayerSize+1)], (hiddenLayerSize, inputLayerSize+1))
        theta2 = np.reshape(nnParams[hiddenLayerSize*(inputLayerSize+1):], (numLabels, hiddenLayerSize+1))
        
        m = X.shape[0]
        
        a1 = np.c_[np.ones(m), X]
        z2 = a1.dot(theta1.conjugate().T)
        a2 = np.c_[np.ones(m), self.sigmoid(z2)]
        z3 = a2.dot(theta2.conjugate().T)
        a3 = self.sigmoid(z3)
        
        #expands y
        y = np.eye(numLabels)[:,y].conjugate().T  
        
        accum1 = 0
        accum2 = 0
        
        for t in range(0,m):
            d3 = a3[t,:] - y[t,:]
            d2 = np.multiply(d3.dot(theta2[:,1-theta2.shape[1]:]), self.sigmoidGradient(z2[t,:]))
            
            accum2 += np.reshape(d3,(-1,1)).dot(np.reshape(a2[t,:], (1,-1)))
            accum1 += np.reshape(d2,(-1,1)).dot(np.reshape(a1[t,:], (1,-1)))

        theta2Grad = accum2/m
        theta1Grad = accum1/m
        
        #regularization
        theta2Grad[:,1-theta2Grad.shape[1]:] += lambdaVal/m * theta2[:,1-theta2.shape[1]:]
        theta1Grad[:,1-theta1Grad.shape[1]:] += lambdaVal/m * theta1[:,1-theta1.shape[1]:]
        
        grad = np.append(theta1Grad.flatten(), theta2Grad.flatten())
        return grad
    
    
    def sigmoidGradient(self, z):
        """
        Computes the element wise sigmoid gradient values given an input matrix
        
        Parameters
        ----------
        z : ndarray
            input for applying the sigmoid gadient function
            
        returns
        -------
        g : ndarray
            output containing the resulting values
        """
        g = np.multiply(self.sigmoid(z), 1.0 - self.sigmoid(z))
        return g 
    