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
T=True
F=False

from inspect import getfullargspec
def securerRepr(obj,ind=1,*args,**xargs):
    try:
        u=obj.__repr__(ind,*args,**xargs)
    except:
        u=obj.__repr__()
    return u

def getAnnotationInit(obj):
    return getfullargspec(obj.__class__.__init__).annotations
def convertCamelToSnake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
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

def merge(source, destination_):
    destination=copy.deepcopy(destination_)
    for key, value in source.items():
        if isinstance(value, dict):
            node = copy.deepcopy(destination.setdefault(key, {}))
            destination[key]=merge(value, node)
        else:
            if key in destination.keys():
                if isinstance(value,list):
                    d=destination[key]
                    destination[key]=value+destination[key]
                elif isNumpyArr(value):
                    destination[key]=np.concatenate((value,destination[key]),axis=0)
                else:
                    destination[key]=[value,destination[key]]
            else:
                destination[key] = value

    return destination
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


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

class ShowWarningsTmp:
    def __enter__(self):
        self.w=getWarnings()
        onWarnings()
    def __exit__(self, type, value, traceback):
        setWarnings(self.w)
class HideWarningsTmp:
    def __enter__(self):
        self.w=getWarnings()
        offWarnings()
    def __exit__(self, type, value, traceback):
        setWarnings(self.w)

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
    if isInt(m):
        return [arr] if m==1 else arr
    if len(m)==1:
        return [arr]
    return arr
def if_exist_global(stri):
    return  ifelse( stri in globals(),lambda:globals()[stri],fnRien)() 
def fnRien(*args,**xargs):None
getsourceP = lambda a:print(getsource(a))