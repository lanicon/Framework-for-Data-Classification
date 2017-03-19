'''
@author: Paul Fung
'''

from scipy.optimize.optimize import fmin_cg

from classifier.BaseClassifier import BaseClassifier
import numpy as np


class LRClassifier(BaseClassifier):
    """
    Logistic Regression Classifier
    """

    def __init__(self, *params):
        """
        Initialization of the Logistic Regression classifier
        
        Parameters
        ----------
        *params : list
            item 1 (optional) - contains the initial value of theta
        """
        super().__init__(*params)
        self.name = "Logistic Regression Classifier"
        self.theta = None
        
        if len(params)==0:
            return
        elif len(params)==1:
            self.theta = params[0]
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
        
        #calculate probabilities of each record belonging to each class
        allP = self.sigmoid(X.dot(self.theta.conjugate().T))
        
        #extracts the index of highest probability for every record
        #i.e. the record belong to the class with highest probabilities (one-vs-all)
        p = allP.argmax(axis=1)
        return p
    
    
    def train(self, update=True, **params):
        """
        Training of model to generate the theta value for the this classifier
        
        Parameters
        ----------
        update : boolean
            when set to true, the classifier's theta value is updated        
        **params : list
            X : ndarray (2D)
                Contains the training set, with each row as one record
            y : ndarray (1D)
                Contains the corresponding label for each row in X
            lambdaVal : float
                Regularization parameter
            maxIter : int
                Maximum number of iterations that the optimization algorithm will run for each label
            numOfLabels : int
                Number of unique labels
                
        Returns:
        --------
        xopt : ndarray (1D)
            optimized theta value
        cost : float
            cost associated to xopt
        """    
        X = params["X"]
        y = params["y"]
        lambdaVal = params["lambdaVal"]
        maxIter = params["maxIter"]
        numOfLabels = params["numOfLabels"]
        
        thetaSize = X.shape[1]
        retTheta=np.zeros((numOfLabels, thetaSize+1))
        X = np.c_[np.ones(X.shape[0]), X]
        theta = np.zeros(thetaSize+1)
        cost = 0
        
        for i in range(0, numOfLabels):
            tmpY = (y==i).astype(int)
            shortCostFunction = lambda theta : self.computeCost(X, tmpY, lambdaVal, theta)
            shortGradFunction = lambda theta : self.computeGradient(X, tmpY, lambdaVal, theta)
            retVal = fmin_cg(shortCostFunction, x0=theta, fprime=shortGradFunction, maxiter=maxIter, full_output=True)
            
            retTheta[i,:] = retVal[0]
            cost += retVal[1]
            
        cost /= numOfLabels
        retVal = (retTheta, cost)
        
        if update:
            self.theta = retTheta
        
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
                Number of unique labels lambdaVals
                
        Returns:
        --------
        lambdaValCost : ndarray (2D)
            costs associated with each item in               
        """            
        X_train = params["X_train"]
        y_train = params["y_train"]
        X_cv = params["X_cv"]
        y_cv = params["y_cv"]
        lambdaVals = params["lambdaVals"]
        maxIter = params["maxIter"]
        numOfLabels = params["numOfLabels"]
        
        lambdaValCost = np.zeros((len(lambdaVals), 2))
        minCost = float("inf")
        minCostLambdaVal = 0
        minCostTheta = None
        X_cv = np.c_[np.ones(X_cv.shape[0]), X_cv]
        
        for i in range(0,len(lambdaVals)):
            print("lambdaVal: ", lambdaVals[i])
            retVal = self.train(update=False, X=X_train, y=y_train, lambdaVal=lambdaVals[i], maxIter=maxIter, numOfLabels=numOfLabels)
            lambdaValCost[i,0] = lambdaVals[i]
            theta_train = retVal[0]
            
            tempCost = 0
            for j in range(0,numOfLabels):
                tmpY = (y_cv==j).astype(int)
                currCost = self.computeCost(X_cv, tmpY, lambdaVals[i], theta_train[j])
                tempCost += currCost
                print("Label: ", j)
                print("currCost: ", currCost)
            
            lambdaValCost[i,1] = tempCost/numOfLabels
            print("AvgCost: ", lambdaValCost[i,1])
            
            if(lambdaValCost[i,1] < minCost):
                minCost = lambdaValCost[i,1]
                minCostTheta = theta_train
                minCostLambdaVal = lambdaVals[i]
                
        print("minCostLambdaVal: ", minCostLambdaVal)
        
        if update:
            self.theta = minCostTheta
        
        return lambdaValCost
        
        
    def computeCost(self, X, y, lambdaVal, theta):
        """
        Computation of cost
        
        Parameters
        ----------
        X : ndarray (2D)
            Contains the training set, with each row as one record
        y : ndarray (1D)
            Contains the corresponding label for each row in X
        lambdaVal : float
            Regularization parameter
        theta : ndarray (1D)
            theta value
            
        Returns
        -------
        J : float
            cost value
        """   
        m = len(y)
        y = np.reshape(y, (-1,1))
        tempTheta = np.append([0],theta[1:])
        tempTheta = np.reshape(tempTheta, (-1,1))
        theta = np.reshape(theta, (-1,1))
        J = 1/m * sum(np.multiply(-y, np.log(self.sigmoid(X.dot(theta)))) - \
                      np.multiply((1-y), np.log(1-self.sigmoid(X.dot(theta))))) + \
                      (lambdaVal/(2*m)) * sum(np.square(tempTheta))
        return J
    
    
    def computeGradient(self, X, y, lambdaVal, theta):
        """
        Computation of gradient
        
        Parameters
        ----------
        X : ndarray (2D)
            Contains the training set, with each row as one record
        y : ndarray (1D)
            Contains the corresponding label for each row in X
        lambdaVal : float
            Regularization parameter
        theta : ndarray (2D)
            theta value
            
        Returns
        -------
        
        grad : float
            gradient value
        """           
        m = len(y)
        y = np.reshape(y, (-1,1))
        tempTheta = np.append([0],theta[1:])
        tempTheta = np.reshape(tempTheta, (-1,1))
        theta = np.reshape(theta, (-1,1))        
        grad = (1/m) * X.conjugate().T.dot(self.sigmoid(X.dot(theta))-y)
        #regularization
        grad += (lambdaVal/m) * tempTheta
        #grad = np.reshape(grad, (1,-1))
        grad = grad.flatten()
        return grad
       
        
    def debugInitWeights(self, thetaSize):
        """
        Fixed initiation of weights
        
        Parameters
        ----------
        thetaSize : int
            size of theta
            
        Returns
        -------
        w : ndarray (1D)
            theta of thetaSize with initialized values (weights)
        """      
        w = np.sin(list(range(1, thetaSize+1)))/10
        return w
    