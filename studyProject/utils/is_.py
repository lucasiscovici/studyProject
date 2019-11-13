import numpy as np
import collections

def isInt(i):
    from .struct import isinstanceBase, isinstance
    return isinstance(i,int)
def isStr(i):
    from .struct import isinstanceBase, isinstance
    return isinstance(i,str)
def isNumpyArr(i):
    from .struct import isinstanceBase, isinstance
    return isinstance(i,np.ndarray)

def isArr(i):
	return isinstance(i, (collections.Sequence, list, tuple, np.ndarray)) and not isStr(i)