'''
@author: Paul Fung
'''

import numpy as np

class BaseClassifier(object):
    """
    Base class for classifiers
    """

    def __init__(self, *params):
        """
        Initialization of the classifier
        
        Parameters
        ----------
        *params : list
            item 1 or more(optional) - contains the initial value(s) of theta
        """
        self.name = None
        
        
    def sigmoid(self, z):
        """
        Computes the element wise sigmoid values given an input matrix
        
        Parameters
        ----------
        z : ndarray
            input for applying the sigmoid function
            
        returns
        -------
        g : ndarray
            output containing the resulting values
        """
        g = np.divide(1.0, np.add(1.0, np.exp(-z)))
        return g
    
    
    def predict(self, features):
        """
        Prediction of label using the input features
        
        Parameters
        ----------
        features : ndarray
            Input variable for predicting its class
        """
        raise NotImplementedError
    
    
    def trainAndTuneLambda(self, update=True, **params):
        """
        Training of model and optimnizing lambda to generate the theta value for the this classifier
        
        Parameters
        ----------
        update : boolean
            when set to true, the classifier's theta value is updated
        **params : list
        """    
        raise NotImplementedError
    
    
    def train(self, update=True, **params):
        """
        Training of model to generate the theta value for the this classifier
        
        Parameters
        ----------
        update : boolean
            when set to true, the classifier's theta value is updated        
        **params : list
        """            
        raise NotImplementedError
    
        
    def computeNumericalGradient(self, computeCost, theta):
        """
        Computation of numerical gradients
        
        Parameters
        ----------
        computeCost : callable, ' 'computeCost(theta)' '
            Cost function for the classifier, theta must be an ndarray (1D)
        theta : ndarray (1D)
            theta value
            
        Returns
        -------
        numGrad : ndarray (1D)
            Numerical gradient values
        """   
        numGrad = list()
        perturbation = np.zeros(len(theta))
        e = 1e-4
        for i in range(0,len(theta)):
            perturbation[i] = e
            loss1 = computeCost(theta - perturbation)
            loss2 = computeCost(theta + perturbation)
            grad = (loss2 - loss1)/(2 * e)
            numGrad.append(grad)
            perturbation[i] = 0
        numGrad = np.reshape(numGrad, (1,-1))
        numGrad = numGrad.flatten()
        return numGrad
    