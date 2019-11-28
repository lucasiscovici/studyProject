import re
import numpy as np
import copy
import warnings
from collections import Counter # Counter counts the number of occurrences of each item
from itertools import tee, count
import string
import random
from functools import wraps
from inspect import getsource
import re
from .is_ import isNumpyArr
from .struct import StudyClass
import pandas as pd
from inspect import getfullargspec
import inspect

T=True
F=False
from .struct import isinstanceBase, isinstance
import sys, os


def createFunctionFromString(string):
    com=compile(string,"<string>","exec")
    gl={}
    ll={}
    eval(com,gl,ll)
    fn=com.co_names[0]
    ggg=ll[fn]
    ggg.__codeString__=string
    return ggg
def indexOfMinForValueInArray(val,arr):
    i=0
    curr=arr[i]
    while curr < val:
        i+=1
        curr=arr[i]
    return i

def setattrAndReturnSelf(obj,k,v):
  setattr(obj,k,v)
  return obj
# Disable
def blockPrint():
    olq=sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return olq

# Restore
def enablePrint(d=None):
    sys.stdout = sys.__stdout__ if d is None else d
    
from contextlib import contextmanager

@contextmanager
def hidePrint():
    oldq=blockPrint()
    yield
    enablePrint(oldq)

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
def get_args(func):
    signature = inspect.signature(func)
    return StudyClass(names=list(signature.parameters.keys()),defaults={
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    },alone=[k for k, v in signature.parameters.items()
        if v.default is inspect.Parameter.empty],signature=signature)

def check_names(l,fn=int):
    return [isNotPossible(lambda:fn(i)) for i in l]

def namesEscape(l,fn=int):
    from . import isArr
    import collections.abc
    e=isinstance(l,collections.abc.Iterable) and not isinstance(l,str)
    #print(e)
    l=l if e else [l]
    l= l.values if isinstance(l,pd.Series) or isinstance(l,pd.DataFrame) else l
    rep=[l[i_] if i else "`"+str(l[i_])+"`" for i_,i in enumerate(check_names(l,fn=fn))]
    return rep if e else rep[0]

def unNamesEscape(l,fn=int):
    from . import isArr
    import collections.abc
    e=isinstance(l,collections.abc.Iterable) and not isinstance(l,str)
    l=l if e else [l]
    # print(type(l))
    l= l.values if isinstance(l,pd.Series) or isinstance(l,pd.DataFrame) else l
    rep= [l[i_] if i else fn(l[i_][1:-1]) for i_,i in enumerate(check_names(l,lambda a:fn(a[1:-1])))]
    return rep if e else rep[0]

def listl(*args):
    return args
def numpyToCatPdSeries(l,argsCat={},argsSeries={}):
    return pd.Series(pd.Categorical(l,**argsCat),**argsSeries)
def isPossible(fn,*args,**xargs):
    possible=True
    try:
        fn(*args,**xargs)
    except:
        possible=False
    return possible
def isNotPossible(fn,*args,**xargs):
    return np.logical_not(isPossible(fn,*args,**xargs))
    
def securerRepr(obj,ind=1,*args,**xargs):
    if (isinstance(obj,dict) or isinstance(obj,tuple) or isinstance(obj,list)) and len(obj)==0:
        return "Empty"
    try:
        u=obj.__repr__(ind,*args,**xargs)
    except:
        try:
            u=obj.__repr__()
        except:
            u="None"
    return u

def getAnnotationInit(obj):
    return getfullargspec(obj.__class__.__init__).annotations
def convertCamelToSnake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def to_camel_case(snake_str):
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + ''.join(x.title() for x in components[1:])
def takeInObjIfInArr(arr,obj):
    return {k:getattr(obj,k) for k in arr}
def getPrivateAttr(obj,private_begin="__",private_end="[^_][^_]",
                       middle=".",private_suffixe=""):
    dd=re.compile("^"+private_begin+private_suffixe+middle+"+"+private_end+"$")
    return {j:getattr(obj,str(j)) for j in [i for i in dir(obj) if dd.match(i)]}

def flatArray(arr):
    return np.array([arr]).flatten()
def getClassName(i):
    try:
        res=i.__name__
    except:
        try:
            res=i.__class__.__name__
        except:
            res=str(type(i))
    return res
def has_method(o, name):
    return name in dir(o)

def merge_dicts(*dict_args,deep=True):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def merge(source, destination_,*,defaults={},add=True):
    # print(source)
    # print(destination_)
    # print(defaults)
    # print(add)
    if defaults is None:
        defaults={}
    destination=copy.deepcopy(destination_)
    destination=removeDefaults(destination,defaults)
    for key, value in source.items():
        if isinstance(value, dict):
            node = copy.deepcopy(destination.setdefault(key, {}))
            try:
                oi=defaults.get(key)
            except Exception as e:
                print(key)
                print(value)
                print(defaults)
                raise e
            destination[key]=merge(value, node,defaults=oi,add=add)
        else:
            if key in destination.keys():
                if isinstance(value,list):
                    d=destination[key]
                    destination[key]=value+destination[key] if add else destination[key]
                elif isNumpyArr(value):
                    destination[key]=np.concatenate((value,destination[key]),axis=0) if add else destination[key]
                else:
                    destination[key]=[value,destination[key]] if add else destination[key]
            else:
                destination[key] = value

    return destination

def removeDefaults(destination,defaults):
    if defaults is None or len(defaults)==0 :
        return destination
    destination2=copy.deepcopy(destination)
    for key, value in destination.items():
        if not isinstance(value, dict):
            if defaults.get(key) is not None:
                if defaults.get(key) == value :
                    del destination2[key]
            else:
                if defaults.get(key) is None and value is None:
                    del destination2[key]
        else:
            f=removeDefaults(value,defaults.get(key))
            if len(f)==0:
                del destination2[key]
    return destination2
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    import string, random
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def iterable(a):
    try:
        (x for x in a)
        return True
    except TypeError:
        return False
def uniquify(arr):
    """Make all the items unique by adding a suffix (1, 2, etc).

    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    seq=arr
    suffs = count(1)
    not_unique = [k for k,v in Counter(seq).items() if v>1] # so we have: ['name', 'zip']
    # suffix generator dict - e.g., {'name': <my_gen>, 'zip': <my_gen>}
    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))  
    for idx,s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            # s was unique
            continue
        else:
            seq[idx] += suffix if int(suffix) > 1 else ""
    return seq

def ifEmpty(arr,default):
    return default if len(arr)==0 else arr

def remove_empty_keys(arr):
    d=arr
    for k in list(d.keys()):
        if not d[k]:
            del d[k]
    return d

def ifelse(cond,ok=None,nok=None):
        return ok if cond else nok
    
def ifelseLenZero(arr,ok,nok):
    return ok if len(arr)==0 else nok


## WARNINGS
def getWarnings():
    return warnings.filters[0][0]

def setWarnings(k):
    warnings.filterwarnings(k)
    
def offWarnings():
    setWarnings('ignore')
    
def onWarnings(d="default"):
    setWarnings('default')

class changeTmpObj:
    def __init__(self,obj,attr,affect=False,returnAttr=False):
        self.obj=obj
        self.attr=attr
        self.returnAttr=returnAttr
        self.v_=getattr(self.obj,self.attr)
        try:
            # print("la")
            if hasattr(getattr(self.obj,self.attr),"clone"):
                # print("ii")
                self.v=getattr(self.obj,self.attr).clone()
            else:
                self.v=copy.deepcopy(getattr(self.obj,self.attr))
        except:
            # print("uddu")
            self.v=copy.copy(getattr(self.obj,self.attr))
        self.affect=affect
    def __enter__(self):
        return self.obj if not self.returnAttr else self.v
    def __exit__(self, type, value, traceback):
        if self.affect:
            self.obj[self.attr]=self.v_
        else:
            try:
                setattr(self.obj,self.attr,self.v_)
            except:
                self.obj[self.attr]=self.v_
class changeTmpVar:
    def __init__(self,obj,attr):
        self.obj=obj
        self.attr=attr
        self.v=self.obj[self.attr]

    def __enter__(self):
        return self.obj

    def __exit__(self, type, value, traceback):
        self.obj[self.attr]=self.v

def fnRien(*args,**xargs):
    return 

def fnReturn(a):
    return lambda *args,**xargs : a

class ShowWarningsTmp:

    def __init__(self):
        self.o=None
    def __enter__(self):
        self.o=warnings.catch_warnings()
        u=self.o.__enter__()
        onWarnings()
        return u
    def __exit__(self, type, value, traceback):
        op=self.o.__exit__(type,value,traceback)
        self.o=None
        return op
class HideWarningsTmp:
    def __init__(self):
        self.o=warnings.catch_warnings()
    def __enter__(self):
        u=self.o.__enter__()
        offWarnings()
        return u
    def __exit__(self, type, value, traceback):
        op=self.o.__exit__(type,value,traceback)
        self.o=warnings.catch_warnings()
        return op

hideWarningsTmp=HideWarningsTmp()
showWarningsTmp=ShowWarningsTmp()

def newStringUniqueInDico(string_,dico_,i_=1,sep_="_"):
        where_=dico_
        name=string_
        nn=name
        i=i_
        while nn in where_:
            nn=name+sep_+str(i)
            i_+=1
        return nn

def removeNone(arr):
    return [i for i in arr if i is not None]

class ContextDecorator(object):
    # __call__ est une méthode magique appelée quand on utilise () sur un objet
    def __call__(self, f):
        # bon, cette partie là suppose que vous savez comment marche un
        # décorateur, si c'est pas le cas, retournez lire l'article sur S&amp;M
        # linké dans le premier paragraphe
        @wraps(f)
        def decorated(*args, **kwds):
            # notez le with appelé sur soi-même, c'est y pas mignon !
            with self:
                return f(*args, **kwds)
        return decorated

## get Methods
def getStaticMethod(mod,cls,met):        
    modu=__import__(mod)
    clss=getattr(modu,cls)
    met=getattr(clss,met)
    return met

def getStaticMethodFromObj(obj,met):
    mod=obj.__module__
    cls=obj.__class__.__name__
    return getStaticMethod(mod,cls,met)

def getStaticMethodFromCls(obj,met):
    mod=obj.__module__
    cls=obj.__name__
    return getStaticMethod(mod,cls,met)

def mapl(*args):
    return list(map(*args))

def zipl(*args):
    return list(zip(*args))

def filterl(*args):
    return list(filter(*args))

def rangel(*args):
    return list(range(*args))
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def if_exist(a):
    return ifelse(a,a)
def ifNotNone(a):
    return ifelse(a is not None,a)
def toTwoDims(arr):
    if np.ndim(arr) == 2:
        return arr
    return np.array(arr).reshape((-1,1))
def toTwoDimsInv(arr):
    if np.ndim(arr) == 2:
        return arr
    return np.array(arr).reshape((1,-1))
def toThreeDimsInv(arr):
    if np.ndim(arr) == 1:
        return np.array(arr).reshape((1,-1))
def ifOneGetArr(arr,m):
    from . import isInt
    if isInt(m):
        return [arr] if m==1 else arr
    if len(m)==1:
        return [arr]
    return arr
def if_exist_global(stri):
    return  ifelse( stri in globals(),lambda:globals()[stri],fnRien)() 
def fnRien(*args,**xargs):None
getsourceP = lambda a:print(getsource(a))