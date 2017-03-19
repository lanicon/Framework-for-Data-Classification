'''
Created on 29 Jan 2017

@author: Paul
'''
import unittest

import numpy as np
import scipy.io as sio
from visualization.DisplayData import displayData

CONST_DATA_PATH = "../../CourseraMLEx3/machine-learning-ex3/ex3/ex3data1.mat"

class Test(unittest.TestCase):


    def testVisualization(self):
        matContent = sio.loadmat(CONST_DATA_PATH)
        features = matContent['X']
        sel = np.matrix(features[3000,:])
        displayData(sel)
        #raw_input("Program paused. Press Enter to continue...")
        #close()
        sel = np.matrix(features[2000,:])
        displayData(sel)
        #raw_input("Program paused. Press Enter to continue...")
        #close()

