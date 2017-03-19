# Mini Framework for Data Classification

This framework is created as a baseline to solve standard data classification problems. It features the use of machine learning algorithms, such as logistic regression and machine learning to classify the [NMIST dataset](http://yann.lecun.com/exdb/mnist/). 

The NMIST dataset is first divided into:

* Training set (60%)
* Cross validation set (20%)
* Test set (20%)

Using the chosen classifier, the ***training set*** is used to create the respective weights by applying optimization to the ***cost*** and ***gradient*** functions.

```python
#anonymous functions for cost and gradient
shortCostFunction = lambda nnParams : self.computeCost(inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, nnParams)
shortGradFunction = lambda nnParams : self.computeGradient(inputLayerSize, hiddenLayerSize, numLabels, X, y, lambdaVal, nnParams)

#optimization
retVal = fmin_cg(shortCostFunction, x0=nnParams, fprime=shortGradFunction, maxiter=maxIter, full_output=True)
```

To regularize for over/underfitting, we varies ***lambdaVal*** and obtain the optimized weights by choosing the one with lowest cost using the ***cross validation set***. 

```python
#iterate through the given lambda values
for i in range(0,len(lambdaVals)):
    print("lambdaVal: ", lambdaVals[i])
    retVal = self.train(update=False, X=X_train, y=y_train, lambdaVal=lambdaVals[i], 
                        maxIter=maxIter, numLabels=numLabels, inputLayerSize = inputLayerSize, 
                        hiddenLayerSize = hiddenLayerSize)
    lambdaValCost[i,0] = lambdaVals[i]
    theta_train = retVal[0]
            
    lambdaValCost[i,1] = self.computeCost(inputLayerSize, hiddenLayerSize, numLabels, X_cv, 
                                          y_cv, lambdaVals[i], theta_train)
    print("currCost: ", lambdaValCost[i,1])
            
    #compare and store the lowest cost, together with the respective weights and lambda value
    if(lambdaValCost[i,1] < minCost):
        minCost = lambdaValCost[i,1]
        minCostTheta = theta_train
        minCostLambdaVal = lambdaVals[i]
                
print("minCostLambdaVal: ", minCostLambdaVal)
```

By using the optimized weights, we compute the model's accuracy using the ***test set***.

```python
#returns all predictions of training set
predictions = classifier.predict(DataStore.test_set_X)
    
#compares predictions with labels
accuracy = np.mean(predictions == DataStore.test_set_y.conjugate().T) * 100
```

In terms of software design, this framework is divided into several packages:

* Data Store
* Classifiers - classification methods developed using OOP
* Visualization
* Test (Unit test)

And the following software design patterns were implemented:

* Dependency injection - swap classification methods without modifying the sourcecode
* Inversion of control - swap classification methods using inheritance and common interface
* Mediator Pattern

Classification methods used are:

* Logistic regression (with one-vs-all method)
* Neural Network


### Program Details

The NMIST dataset consists of hand written digits and their respective labels. 

*Program inputs*
* Features - A handwritten digit is an image made up of 20x20 pixels box, and this gives rise to 400 independent variables (represented by a record/row). There are 5000 images in our dataset (data.mat)
* Labels - The corresponding values to the handwritten digits (data.mat)
* Pre-trained weights of the neural network model (nnWeights.mat)
* Pre-trained weights of the logistic regression model (lrWeights.mat)

*Program outputs*
* Accuracy of the neural network/logistic regression model on the test set
* Recognition of individual handwritten digit from the training dataset using either of the models

<kbd>![picture1](figures/picture1.png)</kbd>

### To run the program

1. Dependencies
	* Python 3 or above
	* Numpy
	* Scipy
	* Matplotlib
	* Pydev

2. After the dependencies are installed, place the project folder into the workspace<br/>
<kbd>![picture2](figures/picture2.png)</kbd>
3. In PyDev, goto 'File -> Import -> General -> Existing Projects into Workspace', then select the file system<br/>
<kbd>![picture3](figures/picture3.png)</kbd><br/>
Click 'Finish'
4. In Package Explorer, double-click on  'MNIST Classification -> main -> Main.py'<br/>
<kbd>![picture4](figures/picture4.png)</kbd>
5. Go to 'Run -> Run As ->  Run Configuration'.<br/>
'Main Module' should be as per the following:<br/>
<kbd>![picture5](figures/picture5.png)</kbd>
6. Go to the Arguments tab, enter ***--NN*** in the 'Program arguments' to use Neural Network for this program run <br/>
(enter ***--LR*** if we want to use Logistic Regression) <br/> By default, the program will run using the pre-trained weights; include ***--T*** to train the model using the training and cross validation sets <br/>
<kbd>![picture6](figures/picture6.png)</kbd><br/>
Click 'run' to execute the program.
7. From the program, the Neural Network classifier has a training set accuracy of 92.4%; this may diff slightly on every program run due to the randomness involved. To continue recognizing the next random image, close the 'Figure 1' window.<br/>
<kbd>![picture7](figures/picture7.png)</kbd>

***Future inclusions***
* More data source types
* More classification models (e.g. SVM, TensorFlow)

***References***
*	Coursera - Stanford University - Machine Learning Course
*	[NMIST Dataset](http://yann.lecun.com/exdb/mnist/)
