'''
@author: Paul Fung
'''

class DataStore(object):
    '''
    Singleton for storing data
    '''
    training_set_X = 0
    training_set_y = 0
    CV_set_X = 0
    CV_set_y = 0
    test_set_X = 0
    test_set_y = 0

    def __init__(self):
        '''
        Constructor
        '''
        