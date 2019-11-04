import numpy as np
def isInt(i):
    from .struct import isinstanceBase, isinstance
    return isinstance(i,int)
def isStr(i):
    from .struct import isinstanceBase, isinstance
    return isinstance(i,str)
def isNumpyArr(i):
    from .struct import isinstanceBase, isinstance
    return isinstance(i,np.ndarray)