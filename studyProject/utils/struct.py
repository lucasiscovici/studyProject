from collections import UserDict,UserList
import numpy as np
from .is_ import isInt, isNumpyArr
# from . import getPrivateAttr, isInt, isNumpyArr, iterable
class StudyList(UserList): pass
import copy
import builtins
isinstanceBase=builtins.isinstance
def isinstance(obj,typ):
    return isinstanceBase(obj,instanceOfType(typ))

class StudyClass:
    def __init__(self,**xargs):
        self.items={}
        for k,v in xargs.items():
            setattr(self,k,v)
    def __setattr__(self, k, v):
        super().__setattr__(k,v)
        if k!="items":
            self.items[k]=v

    def __repr__(self):
        a=""
        for k,v in self.items.items():
            a+="{} :\n".format(k)
            a+="{}\n".format(v)
            a+="\n"
        return a
    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, d): self.__dict__.update(d)
class StudyDict(UserDict,dict):
    def __init__(self, *args, default=None, **kwargs):
        super().__init__(*args, **kwargs)
    def __getitem__(self, key):
        from .util2 import getPrivateAttr
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
            elif hasattr(self,"papa"):
                if hasattr(self,"addPapaIf"):
                    if not self.addPapaIf(rep):
                        return rep
                object.__setattr__(rep,"papa",self.papa)
                if hasattr(self,"attr"):
                    object.__setattr__(rep,"attr",self.attr)
                return rep
            else:
                return rep
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)
    def __reduce__(self):
        return (StudyDict, (), self.__getstate__())
    def __getstate__(self):
        return dict(self)
    def __setstate__(self, state):
        data = state
        self.update(data)  # will *not* call __setitem__
    def __getattr__(self, key):
        from .util2 import getPrivateAttr
        #key=list(self.keys())[key] if isInt(key) else key
        #key=key if isStr(key) else key
        if key.startswith('__') and key.endswith('__'):
            return super().__getattr__(key)
        # print(key)
        if key=="data":
            d=super(UserDict,self).__getattribute__(key)
        else:
            d=self.data
        if key in d:
            rep = d[key]
            atty=getPrivateAttr(self)
            if isinstance(rep,list):
                return studyList(rep,**atty)
            elif isNumpyArr(rep):
                rep=StudyNpArray(rep,**atty)
                return rep
            elif hasattr(self,"papa"):
                if hasattr(self,"addPapaIf"):
                    if not self.addPapaIf(rep):
                        return rep
                object.__setattr__(rep,"papa",self.papa)
                if hasattr(self,"attr"):
                    object.__setattr__(rep,"attr",self.attr)
                return rep
            else:
                return rep
        return super(UserDict,self).__getattribute__(key)


    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, d): self.__dict__.update(d)
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

        
class StudyNpArray(np.ndarray):
    def __new__(cls, array, **kwargs):
        obj = np.asarray(array).view(cls) 
        obj.obj=np.asarray(array)
        for i,j in kwargs.items():
            obj.__setattr__(i,j)
        return obj

    def __array_finalize__(self, obj):
        from .util2 import getPrivateAttr
        if obj is None: return
        kwargs=getPrivateAttr(obj)
        for i,j in kwargs.items():
            self.__setattr__(i,j)
    def __hash__(self):
        return hash(tuple(self))
    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, d): self.__dict__.update(d)     
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
    def __repr__(self,ind=1,ademas=1):
        stri="\n"
        for k,v in self.items():
            stri+="\t"*ind
            stri+=k+" : "
            # stri+="\t"*ind
            try:
                stri+=v.__repr__(ind=ind+ademas)+"\n"
            except:
                stri+=v.__repr__()+"\n"

        return stri[:-1]
class BeautifulList(list):
    def __repr__(self,ind=1):
        from .util2 import iterable
        stri=""
        if not iterable(self):
            return self
        for v in self:
            stri+="\n"
            try:
                stri+=v.__repr__(ind=ind+1)
            except:
                stri+="\t"*(ind)+v.__repr__()

        return stri

from collections import defaultdict
class structClsAuto:
    def __init__(self,arr=[]):
        self.__strucDictAuto__=arr
    def __getattr__(self,x):
        key=x
        if key.startswith('__') and key.endswith('__'):
            return super().__getattr__(key)
        return structClsAuto(self.__strucDictAuto__+[x])
    def __getitem__(self,arrs):
        if isinstance(arrs,list):
            return structClsAuto(self.__strucDictAuto__+[arrs])
    def __eq__(self, other):
        s=dict()
        st=self.__strucDictAuto__[::-1]
        x=st[0]
        st=st[1:]
        s[x]=other
        for k in st:
            u=dict()
            if isinstance(k,list):
                for j in k:
                    u[j]=s
            else:
                u[k]=s
            s=u
        return s
class isinstance2Meta(type):
    def __instancecheck__(self, other):
        return other._instancecheck(self.CLS) if hasattr(other,"_instancecheck") else isinstanceBase(other,self.CLS)
def instanceOfType(typ):
    return isinstance2Meta('SubClass', (), {'CLS': typ})
dicoAuto=structClsAuto()