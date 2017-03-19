'''
@author: Paul Fung
'''

import scipy.io as sio

class BaseDataLoader(object):
    '''
    Base class for data loader
    '''


    def __init__(self):
        pass
        '''
        Constructor
        '''
        
    def loadmat(self, filename):
        raise NotImplementedError
        '''
        Loads matrix from filename
        @param filename: name of file that stores the matrix
        @type string:
        '''
    
    
class MatLoader(BaseDataLoader):
    '''
    Class for loading data from .mat files
    '''


    def __init__(self):
        super().__init__()
        '''
        Constructor
        '''
        
    def loadmat(self, filename):
        '''
        Loads data from .mat file
        '''
        return sio.loadmat(filename)