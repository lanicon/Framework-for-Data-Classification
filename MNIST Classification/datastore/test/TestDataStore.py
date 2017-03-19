'''
@author: Paul Fung
'''

import unittest

from datastore.DataLoader import MatLoader
from datastore.DataStore import DataStore

CONST_DATA_PATH = "../../CourseraMLEx3/machine-learning-ex3/ex3/ex3data1.mat"

class Test(unittest.TestCase):


    def testMatLoader(self):
        matLoader = MatLoader()
        matContent = matLoader.loadmat(filename=CONST_DATA_PATH)
        DataStore.training_set_X = matContent['X']
        DataStore.training_set_y = matContent['y']
        print(len(DataStore.training_set_X[:,1]))
        print(len(DataStore.training_set_X[1,:]))
        print(len(DataStore.training_set_y))
