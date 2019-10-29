from collections import UserDict,UserList
import numpy as np
from . import getPrivateAttr, isInt, isNumpyArr
class StudyList(UserList): pass


class StudyClass:
    def __init__(self,**xargs):
        for k,v in xargs.items():
            setattr(self,k,v)


class StudyDict(UserDict):
    def __getitem__(self, key):
        key=list(self.keys())[key] if isInt(key) else key
        #key=key if isStr(key) else key
        if key in self.data:
            rep = self.data[key]
            atty=getPrivateAttr(self)
            if isinstance(rep,list):
                return studyList(rep,**atty)
            elif isNumpyArr(rep):
                rep=StudyNpArray(rep,**atty)
                return rep
            else:
                return rep
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)
    def __getattr__(self, key):
        #key=list(self.keys())[key] if isInt(key) else key
        #key=key if isStr(key) else key
        if key in self.data:
            rep = self.data[key]
            atty=getPrivateAttr(self)
            if isinstance(rep,list):
                return studyList(rep,**atty)
            elif isNumpyArr(rep):
                rep=StudyNpArray(rep,**atty)
                return rep
            else:
                return rep
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)
        
class StudyNpArray(np.ndarray):
    def __new__(cls, array, **kwargs):
        obj = np.asarray(array).view(cls) 
        obj.obj=np.asarray(array)
        for i,j in kwargs.items():
            obj.__setattr__(i,j)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        kwargs=getPrivateAttr(obj)
        for i,j in kwargs.items():
            self.__setattr__(i,j)
    def __hash__(self):
        return hash(tuple(self))
                           
def studyDico(dico,**args):
    dico=StudyDict(dico)
    for i,j in args.items():
        dico.__setattr__(i,j)
    return dico

def studyList(dico,**args):
    dico=StudyList(dico)
    for i,j in args.items():
        dico.__setattr__(i,j)
    return dico

class Obj(object):
    def __init__(self,**xargs):
        for i,j in xargs.items():
            setattr(self,i,j)

class BeautifulDico(dict):
    def __repr__(self,ind=1):
        stri="\n"
        for k,v in self.items():
            stri+="\t"*ind
            stri+=k+" : "
            # stri+="\t"*ind
            try:
                stri+=v.__repr__(ind=ind+1)+"\n"
            except:
                stri+=v.__repr__()+"\n"

        return stri[:-1]
class BeautifulList(list):
    def __repr__(self,ind=1):
        stri=""
        for v in self:
            stri+="\n"
            try:
                stri+=v.__repr__(ind=ind+1)
            except:
                stri+="\t"*(ind)+v.__repr__()

        return stri