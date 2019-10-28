import functools
import types
import pickle
import dfply as _df
from dfply import X as _X
from sspipe import p as _p, px as _px
def _patch_cls_method(cls, method):
    original = getattr(cls, method)

    @functools.wraps(original)
    def wrapper(self, x, *args, **kwargs):
        if placeholderFn(x) or placeholder(x):
            return NotImplemented
        return original(self, x, *args, **kwargs)

    setattr(cls, method, wrapper)


def patch_cls_operator(cls):
    _patch_cls_method(cls, '__or__')

def patch_all2():
    try:
        import pandas

        patch_cls_operator(pandas.Series)
        patch_cls_operator(pandas.DataFrame)
        patch_cls_operator(pandas.Index)
    except ImportError:
        pass

    try:
        import torch

        patch_cls_operator(torch.Tensor)
    except ImportError:
        pass
T=True
F=False
import pandas as pd
from itertools import combinations
from operator import itemgetter
from sklearn.base import clone,is_classifier,BaseEstimator,ClassifierMixin
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate,check_cv,StratifiedKFold
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
#from sklearn.utils.validation import check_is_fitted
import copy
from matplotlib import colors
from matplotlib.colors import ListedColormap,Normalize
import matplotlib as mpl
import random
import string
from sklearn.exceptions import NotFittedError
from functools import partial
import toolz 
from toolz import curried as toolz_c
from toolz.curried import operator as toolz_c_op
import types
RIEN="___rien___"
from functools import wraps
import dill
import builtins
class SaveLoad:
    @staticmethod
    def load(name,n="rb", compression="lzma",set_default_extension=False,**xargs):
        #return dill.load(open(name,n))
        return compress_pickle.load(name,compression=compression,
                                    set_default_extension=set_default_extension,**xargs)
    
    @staticmethod
    def save(selfo,name,n="wb", compression="lzma",set_default_extension=False,**xargs):
        #return dill.dump(selfo,open(name,n),)
        return compress_pickle.dump(selfo,name,compression=compression,
                                    set_default_extension=set_default_extension,**xargs)
    
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
class toolzFun:
    def __getattr__(self,attr):
        i=[toolz,toolz_c,toolz_c_op]
        for j in i:
            o=getattr(j, attr, None)
            if o:
                return o
        return KeyError
__f=toolzFun()
f__=__f
__ft = ft__=toolz
__fc=fc__=toolz_c
__fcop=fcop=toolz_c_op
class Infix(object):
    def __init__(self, function):
        self.func = function
    def __mod__(self, other):
        return self.func(other)
    def __or__(self, other):
        return self.func(other)
    def __rmod__(self, other):
        return Infix(lambda x, self=self, other=other: self.func(other, x))
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.func(other, x))
    def __call__(self, v1, v2):
        return self.func(v1, v2)
class InfixNumpy(np.ndarray):
    def __new__(cls, function):
        obj = np.ndarray.__new__(cls, 0)
        obj.func = function
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.func = getattr(obj, 'func', None)
    def __mod__(self, other):
        return self.func(___(other))
    def __rmod__(self, other):
        return InfixNumpy(lambda x, self=self, other=other: self.func(other, x))
    def __call__(self, v1, v2):
        return self.func(v1, v2)
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return InfixNumpy(lambda x, self=self, other=other: self.func(other, x))
    
class InfixPandasS(pd.Series):
    @property
    def _constructor(self):
        return InfixPandasS

    @property
    def _constructor_expanddim(self):
        return pd.DataFrame
    
    def __init__(self,function, *args, **kwargs):
        super(InfixPandasS,self).__init__(*args,**kwargs)
        self.func = function
    def __array_finalize__(self, obj):
        if obj is None: return
        self.func = getattr(obj, 'func', None)
    def __mod__(self, other):
        return self.func(___(other))
    def __rmod__(self, other):
        return InfixPandasS(lambda x, self=self, other=other: self.func(other, x))
    def __call__(self, v1, v2):
        return self.func(v1, v2)
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return InfixPandasS(lambda x, self=self, other=other: self.func(other, x))
class InfixPandasD(pd.DataFrame):
    @property
    def _constructor(self):
        return InfixPandasD

    @property
    def _constructor_sliced(self):
        return pd.Series
    
    def __init__(self,function, *args, **kwargs):
        super(InfixPandasD,self).__init__(*args,**kwargs)
        self.func = function
    def __array_finalize__(self, obj):
        if obj is None: return
        self.func = getattr(obj, 'func', None)
    def __mod__(self, other):
        return self.func(___(other))
    def __rmod__(self, other):
        return InfixPandasD(lambda x, self=self, other=other: self.func(other, x))
    def __call__(self, v1, v2):
        return self.func(v1, v2)
    def __or__(self, other):
        return self.func(other)
    def __ror__(self, other):
        return InfixPandasD(lambda x, self=self, other=other: self.func(other, x))

def pipeFn(x,f):
    return f(x)
def pipeRFn(x,f):
    return x(f)
pipe_ = Infix(pipeFn)
pipe__ = InfixNumpy(lambda x,f:f(x,_callMethod=False))
pipe_np = InfixNumpy(pipeFn)
pipe__r = Infix(pipeRFn)
pipe_np_r = InfixNumpy(pipeRFn)
pipe=pipe_np
pipe_r=pipe_np_r
pipe_pds=InfixPandasS(pipeFn)
pipe_pds_r=InfixPandasS(pipeRFn)
pipe_pdd=InfixPandasD(pipeFn)
pipe_pdd_r=InfixPandasD(pipeRFn)
partial_pipe=InfixNumpy(lambda x,f:__f.curry(f)(x))
partial_pipe_r=InfixNumpy(lambda x,f:__f.curry(x)(f))
pipe_map=InfixNumpy(lambda x,f:f__.partial(mapl,f)(x))
pipe_map__=InfixNumpy(lambda x,f:mapl2(f,x))
pipe_mapfn=InfixNumpy(lambda x,f:[f_(x) for f_ in f])
pipe_partial=InfixNumpy(lambda x,f:__f.partial(*f[0],**f[1])(x))
pipe_map_r=InfixNumpy(lambda x,f:f__.partial(mapl,x)(f))
pipe_get=InfixNumpy(lambda arr,i:arr[i])
class FalseClassif(BaseEstimator,ClassifierMixin):
    def __init__(self,y_true,y_pred,w=None):
        self.y_true=y_true
        self.y_pred=y_pred
        self.w=w
    def fit(self,X,y=None):
        return self
                   
    def predict(self,X):
        return self.y_pred
    
    def score(self,X,y):
        return accuracy_score(self.y_true,self.y_pred)
    def __getattr__(self, attr):
        # proxy to the wrapped object
        if self.w is None:
            return super(FalseClassif,self).__getattribute__(attr)
        return getattr(self.w, attr)
import matplotlib.image as mpimg
import tempfile
import os
from pdf2image import convert_from_path
import cairosvg
class TMP_FILE:
    def __init__(self):
        self.i=None
    def get_filename(self,ext="png"):
        _,self.i=tempfile.mkstemp(suffix='.'+ext)
        return self.i
    def delete(self):
        #print(self.i)
        os.remove(self.i)
class IMG:
    def __init__(self,im):
        self.im=im
    def show(self,figure=None,figsize=(20,20),dpi=400,cmap=plt.cm.gray_r):
        img=self.im
        #if not figure:
        #    plt.figure(figsize=figsize,dpi=dpi)
        fig, ax = plt.subplots(figsize=figsize,dpi=dpi)
        ax.axis("off")
        oe=ax.imshow(img,cmap=cmap)
        oe.axes.get_xaxis().set_visible(False)
        oe.axes.get_yaxis().set_visible(False)
        #ax.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
    
    @staticmethod
    def convert(pathBase,pathFin,base,fin):
        if base=="svg" and fin =="png":
            cairosvg.svg2png(url=pathBase, write_to=pathFin)
    @staticmethod
    def getImg(name=None,ext="png",yellow=False,visualizer=None,noSVG=False,**xargs):
        tmpF=TMP_FILE()
        filename=ifelse(name,name,tmpF.get_filename(ext=ext))
        if yellow and visualizer is not None:
            if not noSVG:
                tmpF2=TMP_FILE()
                filename2=tmpF2.get_filename(ext="svg")
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                visualizer.show(outpath=filename2, bbox_inches='tight',pad_inches = 0, dpi=400,format="svg",**xargs)
                plt.close()
                IMG.convert(filename2,filename,"svg",ext)
                tmpF2.delete()
            else:
                visualizer.show(outpath=filename, bbox_inches='tight',pad_inches = 0, dpi=400,format=ext,**xargs)
                plt.close()
        else:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                                #hspace = 0, wspace = 0)
            plt.savefig(filename, bbox_inches='tight',pad_inches = 0,format=ext, dpi=400,**xargs)
            plt.close()
        im=IMG(mpimg.imread(filename))
        tmpF.delete()
        #print(filename)
        return im
    @staticmethod
    def fromPath(p):
        return IMG(mpimg.imread(p))
    
class IMG_GRID:
    @staticmethod
    def grid(imgs,elemsByRows=5, figsize=(12, 12),cmap=plt.cm.gray_r,**xargs):
        instances=imgs
        #instances=instances[:lim]
        images_per_row=elemsByRows
        images_per_row = min(len(instances), images_per_row)
        n_rows = (len(instances) - 1) // images_per_row + 1
        #row_images = []
        #n_empty = n_rows * images_per_row - len(instances)
        
        f, axs2 = plt.subplots(n_rows,images_per_row,figsize=figsize,**xargs)
        axs = np.array(ifOneGetArr(axs2,images_per_row*n_rows)).flatten()
        d=0
        for img, ax in zip(instances, axs):
            #rm=ax.imshow(img.im)
            oe=ax.imshow(img.im,cmap=cmap)
            oe.axes.get_xaxis().set_visible(False)
            oe.axes.get_yaxis().set_visible(False)
            ax.axis('off')
            d+=1
        if d< len(axs):
            #print(axs[d:])
            for ax in axs[d:]:
                #print("la")
                ax.set_axis_off()
                ax.set_visible(False)
        #.set_axis_off()
        #plt.gca().set_aspect('equal', adjustable='datalim')
        #plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #plt.subplots_adjust(bottom=0,right=0)
        plt.show()

class FalseClassifDecFn(BaseEstimator,ClassifierMixin):
    def __init__(self,decValues,y_true,y_pred,w=None):
        self.y_true=y_true
        self.y_pred=y_pred
        self.w=w
        self.decValues=decValues
    def fit(self,X,y=None):
        return self
                   
    def predict(self,X):
        return self.y_pred
    
    def score(self,X,y):
        return accuracy_score(self.y_true,self.y_pred)
    def __getattr__(self, attr):
        # proxy to the wrapped object
        if self.w is None:
            return super(FalseClassifDecFn,self).__getattribute__(attr)
        return getattr(self.w, attr)
    def decision_function(self,X):
        return self.decValues
    
def mapl(*args,**xargs):
    return list(map(*args,**xargs))

def mapl2(f,x):
    f2=__fc.partial(f,_callMethod=F)
    return mapl(f2,x)
def check_is_fitted(i):
    ii=True
    try:
        i.predict([])
    except NotFittedError as e:
        ii=False
    return ii
def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
def mplResetDefault():
    mpl.rcParams.update(mpl.rcParamsDefault)
ddl_heat = ['#DBDBDB','#DCD5CC','#DCCEBE','#DDC8AF','#DEC2A0','#DEBB91',\
            '#DFB583','#DFAE74','#E0A865','#E1A256','#E19B48','#E29539']
ddlheatmap = colors.ListedColormap(ddl_heat)
class placeholderFn:
    def __init__(self,fn):
        self.fn=fn
    def __eq__(self,other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)==other)
    def __ne__(self,other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)!=other)
    def __lt__(self,other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)<other)
    def __gt__(self,other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)>other)
    def __le__(self,other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)<=other)
    def __ge__(self,other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)>=other)
    
    def __add__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)+other)
    def __sub__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)-other)
    def __mul__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)*other)
    def __floordiv__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)//other)
    def __div__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)/other)
    def __mod__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)%other)
    def __pow__(self,other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)**other)
    def __lshift__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)<<other)
    def __rshift__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)>>other)
    def __and__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)&other)
    def __or__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)|other)
    def __ror__(self, other):
        return self(other)
    def __xor__(self, other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)^other)
    def __getattr__(self,a):
        return placeholderFn(lambda selfo,_callMethod=True: getattr(self(selfo,_callMethod=_callMethod),a))
    def __getitem__(self,other):
        return placeholderFn(lambda selfo,**xargs: self(selfo)[other])
    def __call__(self,selfo=None,_callMethod=True,*args,**xargs):
        oo=xargs
        op=args
        o=F
        oo2=lambda selfo:oo
        #print(selfo)
        #print()
        if placeholder.mine(selfo) or placeholderFn.mine(selfo) or (oo.values() |pipe_map| placeholder.mine |pipe| np.any) or (oo.values() |pipe_map| placeholderFn.mine |pipe| np.any):
            o=T
            ou=lambda selfo:[]
            if placeholder.mine(selfo):
                ou=lambda selfoo:[selfoo]
            if placeholderFn.mine(selfo):
                ou=lambda selfoo,selfo=selfo:[selfo(selfoo)]
            selfo=None
            oo2=oo.copy()
            #oo2["selfo"]=selfo
            oo2=lambda selfo,oo=oo2: {k:(selfo if placeholder.mine(v) else ( v(selfo) if placeholderFn.mine(v) else v)) for k,v in oo.items()}
            
            
            
        if (selfo is None) or (isStr(selfo) and selfo==RIEN):
            if o:
                #print(ou(3))
                return placeholderFn(lambda selfo,oo=oo2,ou=ou: self(fakeSelfo(),_callMethod=False)(*ou(selfo),*op,**oo(selfo)))
            return placeholderFn(lambda selfo,oo=oo: (lambda u,oo=oo: u(**oo) if isMethod(u) else u )(self(selfo,_callMethod=False)))
        if _callMethod:
            rep= self.fn(selfo)
        else :
            #print(self.fn)
            try:
                rep=self.fn(selfo,_callMethod=_callMethod)
            except:
                rep= self.fn(selfo)
        #print(rep)
        return rep() if (isMethod(rep) and _callMethod) else rep
    
    @staticmethod
    def mine(i):
        return isinstance(i,placeholderFn)
    
    #def __iter__(self):
    #    return placeholderFn(lambda selfo,**xargs: (self.__dict__[item] for item in self(selfo))
    
    @staticmethod
    def __array_ufunc__(func, method, *args, **kwargs):
        import numpy
        if callable(method) and args[0] == '__call__':
            if method is numpy.bitwise_or:
                if isinstance(args[1], placeholderFn):
                    print(func)
                    print(method)
                    print(args)
                    print(kwargs)
                    return NotImplemented
                else:
                    return args[2](args[1]) if placeholderFn.mine(args[2]) else args[2]
            print(func)
            print(method)
            print(args)
            print(kwargs)
            return NotImplemented
        elif method == '__call__':
            if func.name == 'bitwise_or':
                if isinstance(args[0], placeholderFn):
                    print(func)
                    print(method)
                    print(args)
                    print(kwargs)
                    return NotImplemented
                else:
                    return args[1](args[0]) if placeholderFn.mine(args[1]) else args[1]
            print(func)
            print(method)
            print(args)
            print(kwargs)
            return NotImplemented
        return NotImplemented
    
class placeholder:
    def __init__(self):pass
    def __getattr__(self,a):
        return placeholderFn(lambda selfo,**xargs: getattr(selfo,a))
    def __eq__(self,other):
        return placeholderFn(lambda selfo,**xargs: selfo==other)
    def __ne__(self,other):
        return placeholderFn(lambda selfo,**xargs: selfo!=other)
    def __lt__(self,other):
        return placeholderFn(lambda selfo,**xargs: selfo<other)
    def __gt__(self,other):
        return placeholderFn(lambda selfo,**xargs: selfo>other)
    def __le__(self,other):
        return placeholderFn(lambda selfo,**xargs: selfo<=other)
    def __ge__(self,other):
        return placeholderFn(lambda selfo,**xargs: selfo>=other)
    
    def __add__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo+other)
    def __sub__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo-other)
    def __mul__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo*other)
    def __floordiv__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo//other)
    def __div__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo/other)
    def __mod__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo%other)
    def __pow__(self,other):
        return placeholderFn(lambda selfo,**xargs: selfo**other)
    def __lshift__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo<<other)
    def __rshift__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo>>other)
    def __and__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo&other)
    def __or__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo|other)
    def __xor__(self, other):
        return placeholderFn(lambda selfo,**xargs: selfo^other)
    def __call__(self,other):
        return other
    def __getitem__(self,other):
        return placeholderFn(lambda selfo,**xargs: selfo[other])
    def __ror__(self, other):
        return self(other)
    #def __iter__(self):
    #    return placeholderFn(lambda selfo,**xargs: *selfo) 
    #(self.__dict__[item] for item in sorted(self.__dict__))
    @staticmethod
    def mine(i):
        return isinstance(i,placeholder)
    @staticmethod
    def __array_ufunc__(func, method, *args, **kwargs):
        import numpy
        if callable(method) and args[0] == '__call__':
            if method is numpy.bitwise_or:
                if isinstance(args[1], placeholderFn):
                    print(func)
                    print(method)
                    print(args)
                    print(kwargs)
                    return NotImplemented
                else:
                    return args[2](args[1]) if placeholderFn.mine(args[2]) else args[2]
            print(func)
            print(method)
            print(args)
            print(kwargs)
            return NotImplemented
        elif method == '__call__':
            if func.name == 'bitwise_or':
                if isinstance(args[0], placeholderFn):
                    print(func)
                    print(method)
                    print(args)
                    print(kwargs)
                    return NotImplemented
                else:
                    return args[1](args[0]) if placeholderFn.mine(args[1]) else args[1]
            print(func)
            print(method)
            print(args)
            print(kwargs)
            return NotImplemented
        return NotImplemented
class placeholderOrMe:
    def __init__(self,me):
        self.me=me
        self.isPlaceholderFn=placeholderFn.mine(me)
    def __call__(self,selfo):
        return self.me(selfo) if self.isPlaceholderFn else self.me
__=placeholder()
___=placeholderOrMe
from yellowbrick.classifier import ClassificationReport,ConfusionMatrix,ROCAUC,PrecisionRecallCurve,ClassPredictionError
from yellowbrick import classifier
from yellowbrick.target import ClassBalance as ClassBalance_

class ClassBalance(ClassBalance_):
    def __init__(self, ax=None, labels=None, colors=None, colormap=None,percent=False,ylab="support",
                 fontsize=14,**xargs):
        self.percent=percent
        self.ylab=ylab
        self.fontsize=fontsize
        super(ClassBalance,self).__init__(ax, labels, colors, colormap,**xargs)
    def init2(self,percent=False,ylab="support"):
        self.percent=percent
        self.ylab=ylab
        #super(ClassBalance,self).__init__(*args,**xargs)
    def draw(self):
        if self.percent:
            self.support_=self.support_/self.support_.sum()
        super(ClassBalance,self).draw()
    def finalize(self,**xargs):
        super(ClassBalance,self).finalize(**xargs)
        self.ax.set_ylabel(self.ylab)
        rects=self.ax.containers[0]
        [self.ax.text(rects[a[0]].get_x()+rects[a[0]].get_width()/(2+(len("{:.1f}% ".format(a[1]*50)))*((self.fontsize)/100)), 0.03, "{:.1f}% ".format(a[1]*100),
                                               color='white', va='center',fontsize=self.fontsize, fontweight='bold') for a in enumerate(self.support_)]
def isMethod(m):
    return callable(m)
def fnBase(a,*args,**xargs):
    return a
class studyClassif_viz:
    PrecisionRecallCurve=PrecisionRecallCurve
    ClassPredictionError=ClassPredictionError
    ClassificationReport=ClassificationReport
    ConfusionMatrix=ConfusionMatrix
    ROCAUC=ROCAUC
    confusion_matrix=classifier.confusion_matrix
def visualize_model(X=None, y=None, Xt=None, yt=None, estimator=None,fit=True,names=None,fn=ClassificationReport,
                    cmap="YlGn",size=(600, 360),figsize=(10,10),figure=None,dpi=None,noSVG=False,title="",
                    show=True,beforeShowFn=fnBase,name=None,**kwargs):
    """
    Test various estimators.
    """
    model = estimator

    # Instantiate the classification model and visualizer
    visualizer = fn(
        model, classes=names,
        cmap=cmap, size=size,title=title,is_fitted=not fit, **kwargs
    )
    if name:
        visualizer.name=name
    if fit: visualizer.fit(X, y)
    visualizer.score(Xt, yt)
    beforeShowFn(visualizer)
    img=IMG.getImg(yellow=True,visualizer=visualizer,noSVG=noSVG)
    #if show:visualizer.show()
    if show:img.show(figsize=figsize,dpi=dpi,figure=figure)
    return (visualizer,img)

import re
def getPrivateAttr(obj,private_begin="__",private_end="[^_]",private_suffixe=""):
    dd=re.compile("^"+private_begin+private_suffixe+".+"+private_end+"$")
    return {j:getattr(obj,str(j)) for j in [i for i in dir(obj) if dd.match(i)]}

def new_models(models):
    return [(i,clone(k)) for i,k in models]
def diff_classif_models(models,X_test=None,y_test=None,names=None):
    names=[ i.__class__.__name__ for i in models] if names is None else  names
    kkd=[ y_test!=i.predict(X_test) for i in models]
    combi=list(combinations(range(len(names)),2))
    o=np.zeros((len(names),len(names)))
    o2=np.zeros((len(names),len(names)))
    o3=np.full((len(names),len(names),np.shape(X_test)[0]),-1,dtype="O")
    for (i,j) in combi:
        o[i][j]=(kkd[i]*kkd[j]).sum()
        o2[i][j]=o[i][j]/(kkd[i] | kkd[j]).sum()
        if((o3[i][i]==-1).all()): o3[i][i]=kkd[i]
        if((o3[j][j]==-1).all()): o3[j][j]=kkd[j]
        o3[i][j]=kkd[i]*kkd[j]
        o3[j][i]=kkd[i]*kkd[j]
    return [pd.DataFrame(o,columns=names,index=names),pd.DataFrame(o2,columns=names,index=names),o3,kkd]
def diff_classif_models2(models,X_test=None,y_test=None,names=None):
    #names=[ i.__class__.__name__ for i in models] if names is None else  names
    if np.ndim(y_test)==2:
        raise NotImplemented("MultiLabelNotImplemented")
        #return [diff_classif_models2([ m[:,i] for m in models],X_test,y_test[:,i],names) for i in range(y_test.ndim)]
    kkd=[ y_test!=i for i in models]
    combi=list(combinations(range(len(names)),2))
    o=np.zeros((len(names),len(names)))
    o2=np.zeros((len(names),len(names)))
    o3=np.full((len(names),len(names),np.shape(X_test)[0]),-1,dtype="O")
    for (i,j) in combi:
        o[i][j]=(kkd[i]*kkd[j]).sum()
        o2[i][j]=o[i][j]/(kkd[i] | kkd[j]).sum()
        if((o3[i][i]==-1).all()): o3[i][i]=kkd[i]
        if((o3[j][j]==-1).all()): o3[j][j]=kkd[j]
        o3[i][j]=kkd[i]*kkd[j]
        o3[j][i]=kkd[i]*kkd[j]
    return [pd.DataFrame(o,columns=names,index=names),pd.DataFrame(o2,columns=names,index=names),o3,kkd]

class case(object):
    def __init__(self,j,i,k):
        self.i=i
        self.j=j
        self.k=k
    def __repr__(self):
        return "'[Case] ligne {}, colonne {}, valeur {}'".format(self.i,self.j,self.k)
    def __str__(self):
        return self.__repr__()
    @classmethod
    def fromArr(cls,arr):
        return [cls(i,j,k) for i,j,k in arr]
    
class caseConfusionMat(case):
    def __init__(self,i,j,k,names=None,roundVal=None):
        self.i=i
        self.j=j
        self.roundVal=roundVal
        self.k=k
        self.ni=self.i if names is None else names[self.i]
        self.nj=self.j if names is None else names[self.j]
    def __repr__(self):
        k=ifelse(self.roundVal,np.round(self.k,self.roundVal),self.k)
        return "[CaseConfMat] Il y a {} observations de classe '{}' predite en classe '{}'".format(k,self.ni,self.nj)
    @classmethod
    def fromArr(cls,arr,names=None,noEmpty=True,seuil=0.0,roundVal=None):
        return [cls(i,j,k,names,roundVal=roundVal) for i,j,k in arr if (not noEmpty) or (noEmpty and k>seuil)]
    
class caseConfusionGlobal(case):
    def __init__(self,ie,names=None,typeG=0,roundVal=None):
        i=dict(ie)
        self.typeG=typeG
        self.i=i.items()
        self.roundVal=roundVal
        self.ni=list(i.keys()) if names is None else np.array(names)[list(i.keys())]
    def eachR(self,ni,k):
        typeG=self.typeG
        st=""
        k=ifelse(self.roundVal is None,k,np.round(k,self.roundVal))
        if typeG==0:
            st="[CaseConfGlobal] Classe Actuelle '{}': '{}' mauvaise predictions (prediction de ≠ classes pour cette classe)"
        else:
            st="[CaseConfGlobal] Prediction Classe '{}': '{}' mauvaise predictions (prediction pour d'autres classes)"
        return  st.format(ni,k)
    def __repr__(self,typeG=0):
        return "\n\t".join([self.eachR(self.ni[_i],k) for _i,(i,k) in enumerate(self.i)])
    @classmethod
    def fromArr(cls,arr,names=None,roundVal=None):
        return [cls(i,names,_i,roundVal=roundVal) for _i,i in enumerate(arr)]
    
def most_confused(true,pred,min_val=1,classes=None,
                  lim=100,shuffle=None,percent=False):
        "Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences."
        classes=np.unique(np.concatenate([pred,true]))
        cm = confusion_matrix(true,pred)
        cm2 =cm/np.array(cm).sum(axis=1, keepdims=True) if percent else cm
        np.fill_diagonal(cm2, 0)
        res = [(classes[i],classes[j],cm2[i,j])
                for i,j in zip(*np.where(cm>=min_val))]
        return (sorted(res, key=itemgetter(2), reverse=True)[:lim],len(res)-lim if len(res)>lim else False)
def getObsMostConfused(classe,predit,X,y,pred,lim=10):
    return np.array(X)[(y==classe) & (predit==pred)][:lim]
def most_confused_global(true,pred,classes=None,lim=100,percent=False):
    classes=np.unique(np.concatenate([pred,true]))
    cm = confusion_matrix(true,pred)
    ro=np.sum(cm,axis=1)
    co=np.sum(cm,axis=0)
    cm2 =cm/np.array(cm).sum(axis=1, keepdims=True) if percent else np.sum(cm,axis=1)
    cm3 =cm/np.array(cm).sum(axis=0, keepdims=True) if percent else np.sum(cm,axis=0)
    
    np.fill_diagonal(cm2, 0)
    np.fill_diagonal(cm3, 0)
    lignes=sorted(list(enumerate(np.sum(cm2,axis=1))),key=itemgetter(1), reverse=True)[:lim]
    cols=sorted(list(enumerate(np.sum(cm3,axis=0))),key=itemgetter(1), reverse=True)[:lim]
    
    return ([lignes,cols],len(lignes)-lim if len(cols)>lim else False)

def plot_confusion_matrix3(y,X,j,classes=None,normalize:bool=False, title:str='Confusion matrix', cmap="Blues", slice_size=1,
                              norm_dec:int=2, plot_txt:bool=True, return_fig:bool=None,cv=3,n_jobs=-1, **kwargs):
    plot_confusion_matrix2(y,cross_val_predict(j,X,y,cv=cv,n_jobs=n_jobs),classes=classes,title=title,cmap=cmap,
                           slice_size=1,normalize=normalize,norm_dec=norm_dec,plot_txt=plot_txt,return_fig=return_fig)
def _cross_val_score2(est,train,test,x_test,y_test,name):
    s=[i.score(x_test, y_test) for i in est]
    score2=list(zip(train,test,s))
    df=np.round(pd.DataFrame(score2,columns=["train","validation","test"]),2)
    df.index.name="CV"
    df=pd.concat([df,df.apply(np.mean,axis=0).rename('mean').to_frame().T],axis=0)
    df=pd.concat([df,df.apply(np.std,axis=0).rename('std').to_frame().T],axis=0)
    return (est,(name+"\n"+np.round(df,2).to_string()).replace("\n","\n\t"))

def computeCV(X=None,y=None,cv=5,classifier=True,random_state=42,shuffle=True):
    if isinstance(cv,check_cv2): cv=cv.splited
    elif isInt(cv): cv = check_cv2(X,y,cv,classifier=classifier,random_state=random_state,shuffle=shuffle).splited
    else: cv
    return cv
                    
def cross_val_score2(es,X=None,y=None,x_test=None,y_test=None,cv=5,verbose=False,names=None,predict=False):
    if isinstance(cv,check_cv2): cv=cv.splited
    names=uniquify([i.__class__.__name__ for i in es]) if names is None else names
    es=np.array(es, dtype=object).flatten()
    if(cv==1):
        k=[i.fit(X,y) for i in es]
        d=[_cross_val_score2([i],[i.score(X,y)],[0],x_test,y_test,names[j]) for j,i in enumerate(es)]
        print("\n\n".join([i[1] for i in d]))
        return [i[0] for i in d]
    cvS=[cross_validate(i,X,y,n_jobs=-1,cv=cv,return_estimator=True,return_train_score=True,verbose=verbose) for i in es]
    d=[_cross_val_score2(i["estimator"],i["train_score"],i["test_score"],x_test,y_test,names[j]) for j,i in enumerate(cvS)]
    print("\n\n".join([i[1] for i in d]))
    return [i[0] for i in d]
class check_cv2:
    def __init__(self,X,y,cv_=3,classifier=True,random_state=42,shuffle=True):
        self.cv_=cv_
        self.y=y
        self.X=X
        self.classifier=classifier
        self.random_state=random_state
        self.shuffle=shuffle
        self.splited=self.split()
    def split(self,*args):
        e=check_cv(self.cv_,self.y,classifier=self.classifier)
        e.shuffle=self.shuffle
        e.random_state=self.random_state
        return list(e.split(self.X,self.y))
def removeTrimEmptyStr(s):
        return [i.strip() for i in s if len(i)>0]
def plot_classification_report(cr, title=None, cmap=plt.cm.gray_r,figsize=(6,5),fontsize=13,
                               col1="w",col2="k",titleSize=10,show=True,noText=False,rotationXLabel=45,vmin=None,vmax=None,onlyCr=False,sameColorBar=False
                               ,**xargs):
    title = title or 'Classification report'
    lines = cr.split('\n')
    classes = []
    matrix = []
    for line in lines[2:(len(lines)-1)]:
        s = removeTrimEmptyStr(line.split("    "))
        if len(s)==0: continue
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)
    if onlyCr:
        return matrix
    fig, ax = plt.subplots(1,figsize=figsize)
    thresh=(np.max(matrix)-np.min(matrix))/2.+np.min(matrix)
    #print(thresh)
    if sameColorBar:
        thresh=(vmax-vmin)/2.+vmin
    dn=Normalize(vmin=vmin,vmax=vmax)
    for column in range(len(matrix[0])):
        for row in range(len(classes)):
            txt = matrix[row][column]
            #print(txt)
            if not noText:
                ax.text(column,row,"{:.1f}%".format(matrix[row][column]*100),va='center',ha='center',
                        fontsize=fontsize,color=frontColorFromCmapAndValue(dn(txt),cmap))
    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap,vmin=vmin,vmax=vmax)
    plt.title(title,fontsize=titleSize)
    cbar=plt.colorbar()
    cbar.ax.set_yticklabels([str(int(float(t)*100)) for t in cbar.ax.get_yticks()])
    x_tick_marks = np.arange(len(matrix[0]))
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=rotationXLabel)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    #plt.show()
    img=IMG.getImg()
    if show:img.show(figsize=figsize)
    return img
                           
def cross_fn_cv(fn,cv,X=None,y=None):
    if isinstance(cv,check_cv2):
        if X is None:
            X = cv.X
        if y is None:
            y = cv.y

        vc=cv.split()
    else:
        assert X is not None 
        assert Y is not None
        vc=cv
    return [fn(X[i],y[i],X[j],y[j]) for (i,j) in vc] 
from collections import Counter # Counter counts the number of occurrences of each item
from itertools import tee, count

def uniquify(seq2):
    """Make all the items unique by adding a suffix (1, 2, etc).

    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    seq=seq2
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
            seq[idx] += suffix
    return seq
def zipl(*args):
    return list(zip(*args))
def filterl(*args):
    return list(filter(*args))
def rangel(*args):
    return list(range(*args))
def ifEmptySet(s,li):
    return s if len(li)==0 else li
def remove_empty_keys(d):
    for k in list(d.keys()):
        if not d[k]:
            del d[k]
    return d
def ifelse(cond,ok=None,nok=None):
        return ok if cond else nok
def ifelseLen(m,ok,nok):
    return ifelse(len(m)==0,ok,nok)
import warnings
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
class modErr:
    def __init__(self,arr,names,noEmpty=False):
        self.arr=arr
        self.names=names
        self.noEmpty=noEmpty
    def printM(self,arr,m):
        if self.noEmpty and len(arr)==0: return None
        a=str(m)
        a+="\t"+np.array2string(arr,separator=", ")
        return a
    def __repr__(self):
        return "\n".join(removeNone([self.printM(np.where(j)[0],self.names[i]) for i,j in enumerate(self.arr)]))

class loadFnIfCallOrSubset(object):
    def __init__(self,selfo,fn):
        self.fn=fn
        self.selfo=selfo
    def __getitem__(self,i):
        resu=self.fn(self.selfo)
        return resu[i]
    def load(self,selfo):
        return self.fn(selfo)
    @staticmethod
    def mine(l):
        return isinstance(l,loadFnIfCallOrSubset)
class tg:
    def __init__(self,i):
        self.i=i
    def __repr__(self):
        return ""
class mostConfGlobL:
    def __init__(self,d,lignes=True):
        self.lignes=lignes
        self.d=d
    def __repr__(self):
        n="Actuelle" if self.lignes else "Predite"
        oo=self.d.items()
        jj=[]
def printArr(arr,noPrint=False):
    stre="\n".join([ "\t"+str(i) for i in arr])
    if not noPrint: print(stre)
    return arr if not noPrint else stre
def printMostConfGlob(arr,noPrint=False):
    ne=["ACTUELLE","PRÉDITE"]
    stre="\n".join([ "\n\t"+ne[_i]+"\n\t"+str(i) for _i,i in enumerate(arr)])
    if not noPrint: print(stre)
    return arr if not noPrint else stre
def fromStudy(s,s2,models=None):
    s.cvFromOtherStudy(s2)
    if models is not None and ((isinstance(models,bool) and models==True) or (not isinstance(models,bool))):
        s.addModelsToCurrentCV_(models)
    return s
def computeStudy(s,cv=3,name=None,models=None,returnSelf=True,**xargs):
    name=ifelse(name,name,s.id_+"_cv"+str(cv))
    if models is not None:
        return s.addModelsToCurrentCV(models,returnSelf=returnSelf)
    s.computeCV(cv=cv,name=name,**xargs)
    return s
def computeOrSave(s,s2=None,computeKey="c",auto=False,models=None,**xargs):
    if not auto:
        a = input()
    else:
        a=ifelse(s2,"s","c")
    if a==computeKey:
        return computeStudy(s,models=models,**xargs)
    if s2 is not None:
        return fromStudy(s,s2,models=models)
    raise KeyError("mauvaise key ou par de s2 (sauvegarde)")
def if_exist(a):
    return ifelse(a,a)
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

def computeOrSaveAuto(s,prefix="",suffix="_",auto=F,affectGlobals=True,returnOK=False,**xargs):
    n=prefix+s.id_+suffix
    resu=computeOrSave(s,if_exist_global(n),auto=auto,**xargs)
    if affectGlobals:
        globals()[n]=resu
    return ifelse(returnOK,resu)
    
def printDico(dico,fn=lambda a,*args,**xargs:a,moreAdd=None,more="\n(+ {})"):
    moreAdd = [False]*len(dico.keys()) if moreAdd is None else  moreAdd
    oo=[ i+"\n"+fn(j,noPrint=True) for i,j in dico.items()]
    oo=[i+more.format(moreAdd[_i]) if moreAdd[_i] else i for _i,i in enumerate(oo)]
    print("\n\n".join(oo))
    return dico
def isInt(i):
    return isinstance(i,int)
def isStr(i):
    return isinstance(i,str)
def isNumpyArr(i):
    return isinstance(i,np.ndarray)
def ifOkDoFn(i,ok,fn):
    return fn(i) if ok else i            
def flatArray(arr):
    return np.array([arr]).flatten()
def getClassName(i):
    try:
        res=i.__class__.__name__
    except:
        res=None
    return res
def has_method(o, name):
    return name in dir(o)
def merge(source, destination_):
    """
    run me with nosetests --with-doctest file.py

    >>> a = { 'preds' : { 'Tr' : { 'original' : [1], 'sorted' : [1] } } }
    >>> b = { 'preds' : { 'Tr' : { 'original' : [2], 'sorted' : [2] } } }
    >>> merge(a, b) == { 'first' : { 'all_rows' : { 'pass' : 'dog', 'fail' : 'cat', 'number' : '5' } } }
    True
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    { 'Tr' : { 'original' : [1], 'sorted' : [1] } }
    { 'Tr' : { 'original' : [2], 'sorted' : [2] } }
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    { 'original' : [1], 'sorted' : [1] }
    { 'original' : [2], 'sorted' : [2] }
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
     [1,2]
    
    
    """
    destination=copy.deepcopy(destination_)
    for key, value in source.items():
        #print(key)
        if isinstance(value, dict):
            # get node or create one
            node = copy.deepcopy(destination.setdefault(key, {}))
            destination[key]=merge(value, node)
        else:
            if key in destination.keys():
                if isinstance(value,list):
                    #print(np.shape(destination[key]))
                    d=destination[key]
                    #print(isNumpyArr(d))
                    #d= d.tolist() if isNumpyArr(d) else d
                    destination[key]=value+destination[key]
                elif isNumpyArr(value):
                    #print(key)
                    
                    #print(np.shape(value))
                    #print(np.shape(destination[key]))
                    
                    destination[key]=np.concatenate((value,destination[key]),axis=0)
                else:
                    destination[key]=[value,destination[key]]
            else:
                destination[key] = value

    return destination
class StudyNpArray(np.ndarray):
    def __new__(cls, array, **kwargs):
        obj = np.asarray(array).view(cls) 
        obj.obj=np.asarray(array)
        for i,j in kwargs.items():
            obj.__setattr__(i,j)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        kwargs=getPrivateAttr(obj,private_suffixe="study")
        for i,j in kwargs.items():
            self.__setattr__(i,j)
    def __hash__(self):
        return hash(tuple(self))
                           
def StudyDico(dico,**args):
    dico=StudyClassifDict(dico)
    for i,j in args.items():
        dico.__setattr__(i,j)
    return dico
def StudyList(dico,**args):
    dico=StudyClassifList(dico)
    for i,j in args.items():
        dico.__setattr__(i,j)
    return dico
from collections import UserDict,UserList
class StudyClassifDict(UserDict):
    def __getitem__(self, key):
        key=list(self.keys())[key] if isInt(key) else key
        #key=key if isStr(key) else key
        if key in self.data:
            rep = self.data[key]
            atty=getPrivateAttr(self,private_suffixe="study")
            if isinstance(rep,list):
                return StudyList(rep,**atty)
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
            atty=getPrivateAttr(self,private_suffixe="study")
            if isinstance(rep,list):
                return StudyList(rep,**atty)
            elif isNumpyArr(rep):
                rep=StudyNpArray(rep,**atty)
                return rep
            else:
                return rep
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)
def getDecFn(l,X):
    attrs = ("predict_proba", "decision_function")

    # Return the first resolved function
    for attr in attrs:
        try:
            method = getattr(l, attr, None)
            if method:
                return method(X)
        except AttributeError:
            # Some Scikit-Learn estimators have both probability and
            # decision functions but override __getattr__ and raise an
            # AttributeError on access.
            # Note that because of the ordering of our attrs above,
            # estimators with both will *only* ever use probability.
            continue
    raise KeyError
import sklearn 
def luminence(t):
    t=t|pipe_map| __*255
    luminance_ =  (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2])/255
    return luminance_
def frontColorFromCmapAndValue(value,cmap,c1="k",c2="w"):
    return c1 if luminence(cmap(value)) > 0.5 else c2
isNone=lambda a:a is None
isNotNone=lambda a:a is not None
def get_metric(me):
    if hasattr(sklearn.metrics,me):
        return getattr(sklearn.metrics,me)
    elif hasattr(sklearn.metrics,me+"_score"):
        return getattr(sklearn.metrics,me+"_score")
    raise KeyError(me)
from sklearn.metrics import accuracy_score,fbeta_score,get_scorer
class StudyClassifList(UserList): pass
class StudyClassif:
    def __init__(self,models=None,X_test=None,y_test=None,
                 X_train=None,y_train=None,
                 namesCls=None,metric="accuracy",
                id_=None,fromGlobal=None):
        #offWarnings()
        self.fromGlobal=fromGlobal
        self._y_train=None
        self._X_train=None
        self._y_test=None
        self._X_test=None
        self._models=None
        self._names=None
        self.isReady=False
        self.processingDataFn = None
        self.isProcess=False
        self.id_=self.id=ifelse(id_,id_,randomString())
        self.history={}
        self._cv={}
        self.metric=metric
        self._nameCV=None
        self.isMultiLabel=False
        self.setDataTest(X_test,y_test)
        self.setDataTrain(X_train,y_train)
        self.setNamesCls(namesCls)
        if models is not None: self._init(models)
        self.isReady=self.checkIsReady()
    
    def finalCheck(self):
        if self._diff is None:
            self._diff=loadFnIfCallOrSubset(self,self._diffFn)
            
    def checkIsReady(self):
        g=[self._y_train,
            self._X_train,
            self._y_test,
            self._X_test,
            self._models,
            self._names]
        return g |pipe_map| isNotNone |pipe| np.all
    
    def ifNotReady(self,fn):
        if not self.isReady:
            ___(fn)(self)
    @staticmethod
    def _diffFn(self):
        self._diff=self.diff_classif_models()
        return self._diff
    def _init(self,models):
        self._models=flatArray([models])
        self._names=np.array(uniquify([getClassName(i) for i in self._models]))
        self._inames=np.array(rangel(len(self._models)))
        self._namesi=dict(zip(self._names,self._inames))
        self._diff=loadFnIfCallOrSubset(self,self._diffFn)
        self.isReady=self.checkIsReady()
    
    def setDataXY(self,X,y,test_size=0.2,shuffle=True,random_state=42):
        X=___(X)(self)
        y=___(y)(self)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,shuffle=shuffle,random_state=random_state)
        self.setDataX(X_train,X_test)
        self.setDataY(y_train,y_test)
        self.isReadyData=True
        self.isReady=self.checkIsReady()
                           
    def setDataTrainTest(self,X_train=None,X_test=None,y_train=None,y_test=None,names=None,id_=None):
        if id_ is None and np.any(mapl(lambda a:a is None,[X_train,X_test,y_train,y_test])):
           raise KeyError("if id_ is None, all of [X_train,X_test,y_train,y_test] must be specified  ")
        if id_ is not None and self.fromGlobal is None:
            raise KeyError("if id_ is specified, fromGlobal must be set")
        if id_ is not None and id_ not in self.fromGlobal.data:
            raise KeyError("id_ not in global")
        if id_ is not None:
            X_train,X_test,y_train,y_test,names=self.fromGlobal.data[id_]
            self.idData=id_
        X_train=___(X_train)(self)
        y_train=___(y_train)(self)
        X_test=___(X_test)(self)
        y_test=___(y_test)(self)
        self.setDataX(X_train,X_test)
        self.setDataY(y_train,y_test,names=names)
        self.isReady=self.checkIsReady()
    
    def preprocessingData(self,fn):
        if fn is None: return
        self.processingDataFn = fn
        self.isProcess=True
        X_train,X_test,y_train,y_test,namesCls=fn(self.X_train,self.X_test,self.y_train,self.y_test,self.namesCls)
        self.setDataTrainTest(X_train,X_test,y_train,y_test,namesCls)
    
    def setDataY(self,y_train,y_test,names=None):
        self._y_train=___(y_train)(self)
        self._y_test=___(y_test)(self)
        self.isMultiLabel=np.ndim(self._y_train) > 1
        if self.isMultiLabel:raise NotImplemented("MultiLabelNotImplemented")
        if names is not None: 
            self.setNamesCls(names)
        self.isReady=self.checkIsReady()
            
    def setDataTrain(self,X_train,y_train,names=None):
        self._X_train=___(X_train)(self)
        self._y_train=___(y_train)(self)
        if names is not None: 
            self.setNamesCls(names)
        self.isReady=self.checkIsReady()
            
    def setDataTest(self,X_test,y_test,names=None):
        self._X_test=___(X_test)(self)
        self._y_test=___(y_test)(self)
        if names is not None: 
            self.setNamesCls(names)  
        self.isReady=self.checkIsReady()
            
    def setDataX(self,X_train,X_test):
        self._X_train=___(X_train)(self)
        self._X_test=___(X_test)(self)
        self.isReady=self.checkIsReady()

    def setNamesCls(self,names):
        self._namesCls=names
        self.isReady=self.checkIsReady()
    
    def setModels(self,models):
        self._init(models)
        self.isReady=self.checkIsReady()
                           
    def getIndexFromNames(self,arr,returnOK=True):
        return [self.namesi[i] if isStr(i) else i for i in arr] if returnOK else None
                           
    def diff_classif_models(self,test=None, returnOK=True):
        test = (np.all([check_is_fitted(i) for i in self.models])) if test is None else test
        
        o=diff_classif_models(self.models,self.X_test,self.y_test,self.names) if test else diff_classif_models2(self.getCvPreds(),self.X_train,self.y_train,self.names) 
        return o if returnOK else None                  
    def most_confused(self,m=[],min_val=1,noEmpty=T,classNames=True,lim=100,shuffle=False,
                     globally=False,returnOK=False,percent=False,roundVal=None):
        if not globally:
            offWarnings()
            res= {i+" (N°"+str(o)+")" :most_confused(self.y_train,j,min_val=min_val,lim=lim,shuffle=shuffle,percent=percent)  for o,(i,j) in enumerate(zipl(self.names,self.getCvPreds()))}
            moreS=[j[1] for i,j in res.items()]
            res={i:caseConfusionMat.fromArr(j[0],self.namesCls if classNames else None ,noEmpty,roundVal=roundVal) for i,j in res.items() }
            onWarnings()
            res=remove_empty_keys(res) if noEmpty else res
            o= printDico(res,fn=printArr,moreAdd=moreS)
        else:
            allPred=[[self.y_test,j.predict(self.X_test)]  for o,(i,j) in enumerate(zipl(self.names,self.models))]
            allPredN=np.hstack(allPred)
            res=most_confused(allPredN[0],allPredN[1],min_val=min_val,lim=lim,shuffle=shuffle) 
            moreS=[res[1]]
            res={"globally":caseConfusionMat.fromArr(res[0],self.namesCls if classNames else None )}
            res=remove_empty_keys(res) if noEmpty else res
            o= printDico(res,fn=printArr,moreAdd=moreS)
        return o if returnOK else None 
    
    def _plot_confusion_matrix2(self,true,pred,classes=None,normalize:bool=False, title:str='Confusion matrix', cmap="Blues", slice_size=1,
                              norm_dec:int=2, plot_txt:bool=True, return_fig:bool=None,justCM=False,normalizeByRows=False,
                              diagZero=False,colorbar=False,noText=False,long2=0,long=50,linewidth=5,alpha=0.5,modulo=10,figsizeShow=(4,4),fontsize=12,osefLine=False,titleSize=10,vmin=None,vmax=None,roundVal=None,figsize=(6,6),both=False,col1="white",col2="black",**kwargs):
        "Plot the confusion matrix, with `title` and using `cmap`."
        #row_sums = conf_mx.sum(axis=1, keepdims=True)
        # This function is mainly copied from the sklearn docs
        if classes is None:
            classes = np.unique(np.concatenate([true,pred]))
        cm = confusion_matrix(true,pred)
        cmi=cm
        cm_sum = cm.sum(axis=1, keepdims=True)
        if normalize or normalizeByRows: 
            cm = cm/cm_sum
            cmi_perc=cm
            cm=cm* 100
        cm2=np.copy(cm)
        cm22=np.copy(cm2)
        #print(cmap(0)[0])
        #print(99.9 if cmap(0)[0]==0 else 0)
        if diagZero: np.fill_diagonal(cm22,0)
        if justCM:
            return cm22
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cmi[i, j]
                p = cm2[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)  
                    
        #if normalize or normalizeByRows: cm = (cm.astype('float') / cm.sum(axis=1))[:, np.newaxis]
        if roundVal is not None: cm=np.round(cm,roundVal)
        fig = plt.figure(figsize=figsize,**kwargs)
        if vmin is not None:
            #f1=1-cm2
            #if diagZero: np.fill_diagonal(f1,0 if cmap(0)[0]==0 else 99.9)
            #print(cmap(0)[0]==0)
            #print(cm2)
            #d=Normalize(vmin=vmin/100. if cmap(0)[0]==0 else 0,vmax=0.999 if cmap(0)[0]==0 else max(vmax/100.,99.9) )
            #print(vmin/100. if cmap(0)[0]==0 else 0)
            #print(0.999 if cmap(0)[0]==0 else max(vmax/100.,99.9) )
            #cm2=minmax_scale(f1,(vmin/100. if cmap(0)[0]==0 else 0,0.999 if cmap(0)[0]==0 else max(vmax/100.,99.9)))
            #print(cm2)
            plt.imshow(cm22, cmap=cmap,vmin=vmin,vmax=vmax)
        else:
            plt.imshow(cm22,cmap=cmap)
        if colorbar: 
            cbar=plt.colorbar()
            #cbar.ax.set_yticklabels([str(int(float(t))) for t in cbar.ax.get_yticks()])
        plt.title(title,fontsize=titleSize)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes, rotation=0)
        vmin = ifelse(vmin,vmin,cm2.min())
        vmax = ifelse(vmax,vmax,cm2.max())
        
        dn=Normalize(vmin=vmin,vmax=vmax)
        if plot_txt:
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                coeff = ("{:.2f}%".format(cm[i, j]*100.0) if normalize or normalizeByRows else f'{cm[i, j]}') if not both else annot[i,j]
                #value = cm_display[x, y]
                #svalue = "{:0.0f}".format(value)
                #coeff="{:0.0f}%".format(cm2[i, j])
                if not noText:
                    plt.text(j, i, coeff, horizontalalignment="center",
                         verticalalignment="center", 
                         color=frontColorFromCmapAndValue(dn(cm22[i, j]),cmap),fontsize=fontsize)
        
        #plt.tight_layout()
        plt.ylabel('Actuelle')
        plt.xlabel('Predite')
        plt.grid(False)
        sdd=IMG.getImg()
        if osefLine:
            fig, ax = plt.subplots(figsize=figsizeShow,dpi=400)
            ax.axis("off")
            ax.imshow(sdd.im)
            x=range(long,sdd.im.shape[0]+long2)
            #x2=range(sdd.im.shape[0])
            ax.plot(np.array(list(x))-modulo, x, '--', linewidth=linewidth, color='firebrick',alpha=alpha)
            sdd=IMG.getImg()
        return (sdd,cm2)
        #fig
        
    def plotConfusionMatrix(self,
                            m=[],
                            test=False,
                            normalizeByRows=False,cmap=plt.cm.Blues_r,cmap_errors=plt.cm.gray,
                            diagZero=False,colorbar=False,roundVal=None,errors=False,col1="black",
                            col2="white",show=True,returnOK=False,modulo=-120,figsize=(2,2),figsizeShow=(1.5,1.5),osefLine=False,both=False,sameColorBar=False,**xargs):
        if both and not errors:
            warnings.warn("both True = errors True")
            errors=True
        if errors:
            yellowBrick=F
            normalizeByRows=True
            diagZero=True,
            colorbar=T
            roundVal=2
            cmap=cmap_errors
            col1="k"
            col2="w"
            sameColorBar=True
            osefLine=True
        
        zizi = self.m_process_test(m)
        _,y_true=self.m_process_true(m)
        vmin=None
        vmax=None
        if sameColorBar:
            vmin,vmax =[
                self._plot_confusion_matrix2(y_true,y_pred,title=name,classes=self.namesCls,
                                                                         normalizeByRows=normalizeByRows,cmap=cmap,
                                                                        diagZero=diagZero,colorbar=colorbar,roundVal=roundVal,
                                                                       col1=col1,col2=col2,figsize=figsize,justCM=True,both=both,**xargs)
                for index,(name,y_pred) in zizi
                ] |pipe_mapfn| [np.min,np.max]
        #print([vmin,vmax])
        zizi = self.m_process_test(m)
        _,y_true=self.m_process_true(m)
        o2=[
            self._plot_confusion_matrix2(y_true,y_pred,title=name,classes=self.namesCls,
                                                                     normalizeByRows=normalizeByRows,cmap=cmap,
                                                                    diagZero=diagZero,colorbar=colorbar,roundVal=roundVal,
                                                                   col1=col1,modulo=modulo,col2=col2,figsizeShow=figsizeShow,osefLine=osefLine,vmin=vmin,vmax=vmax,figsize=figsize,both=both,**xargs)
            for index,(name,y_pred) in zizi
          ]
        #print(o2)
        if show:
            o2 |pipe_map| fc__.get(0) |pipe_map| __.show(figsize=figsizeShow,dpi=400,cmap=cmap) 
            #o2 %pipe% f__.partial(mapl,__.show(figsize=(figsize[0]+1,figsize[1]+1)))
        
        return ifelse(returnOK,o2)
        #studyClassif_viz.confusion_matrix(self.createFalseClassif())
    
                           
    def getModelsErrors(self,m=[],noEmpty=False,returnOK=False):
        o3=self.diff[2]
        mm=flatArray(m)
        mm=self.getIndexFromNames(mm)
        o33=[o3[i][i] for i in range(np.shape(o3)[0])]
        o= modErr(o33 if len(mm) == 0 else np.array(o33)[mm] ,self.names if len(mm) == 0 else self.names[mm],noEmpty=noEmpty)
        return ifelse(returnOK,o)
    
    def getModelsSameErrors(self,m=[],pct=False,noEmpty=False,returnOK=True):
        o3=self.diff[0]
        mm=flatArray(m)
        mm=self.getIndexFromNames(mm)
        if pct:
            o3 = self.diff[1]
        o=o3 if len(mm)==0 else o3.iloc[mm,mm]
        return ifelse(returnOK,o)
                           
    def cross_val_score2(self,m=[],cv=3,verbose=0,returnNewStudy=False,returnOK=False,predict=False):
        X_train=self.X_train
        y_train=self.y_train
        mm=flatArray(m)
        mm=self.getIndexFromNames(mm)
        cv = cross_val_score2(self.models if len(mm)==0 else self.models[mm],
                              X_train,y_train,
                              self.X_test,self.y_test,
                              cv=cv,verbose=verbose,
                              names=self.names if len(mm)==0 else self.names[mm],
                             predict=predict)
        if returnNewStudy: 
            return ifelse(returnOK,(cv,self.updateModels(cv)),self.updateModels(cv))
        else : 
            return ifelse(returnOK,cv)
                           
    def updateModels(self,models,id_=None,history=True):
        #me=copy.deepcopy(self)
        me=StudyClassif(id_=id_)
        #me.history=self.history
        #me.history[self.id_]=self
        me.setDataTrainTest(self.X_train,self.X_test,
                            self.y_train,self.y_test,
                            names=self.namesCls)
        me.setModels(models)
        return me

    def isModelsFitted(self):
        return np.all([check_is_fitted(i) for i in self.models])
        
    def classification_report(self,m=[],returnOK=False,printOK=True,test=None,**xargs):
        _,y_true=self.m_process_true(m)
        od={name: classification_report(y_true,y_pred,target_names=self.namesCls)
               for index,(name,y_pred) in self.m_process_test(m,test=test)}
        o=ifelse(printOK,lambda:printDico(od),lambda:od)()
        return ifelse(returnOK,o)
                           

    def plot_classification_report(self,m=[],title=None,returnOK=False,test=None,
                                   figsize=(6,5),yellowBrick=False,show=True,sameColorBar=True,**xargs):
        if yellowBrick:
            plt.figure(figsize=figsize);
            _,y_true=self.m_process_true(m)
            o=[self.visualize_model(self.createFalseClassif(y_true,y_pred,index),
                                  fit=False,fn=studyClassif_viz.ClassificationReport,title=name,show=show,figsize=figsize,**xargs) 
             for index,(name,y_pred) in self.m_process_test(m)]
            return ifelse(returnOK,o)
        od=self.classification_report(m=m,returnOK=True,printOK=False,test=test)
        vmin=None
        vmax=None
        o2=[plot_classification_report(j,i,onlyCr=True) for i,j in od.items()]
        if sameColorBar:
            vmin=np.min([np.min(i) for i in o2])
            vmax=np.max([np.max(i) for i in o2])
        o=[plot_classification_report(j,i,figsize=figsize,show=show,vmin=vmin,vmax=vmax,sameColorBar=sameColorBar,**xargs) for i,j in od.items()]                
        return ifelse(returnOK,o)
    
    def m_process_test(self,m,test=None,enumerate_=T,predict=True,decVal=False):
        mm=flatArray(m)
        mm=self.getIndexFromNames(mm)
        test=ifelse(test,test,self.isModelsFitted())
        if not decVal:
            zizi= zipl(ifelseLen(mm,self.names,
                                    self.names[mm]),
                        ifOneGetArr(ifelseLen(mm,lambda:self.getCvPreds(),
                                     lambda mm=mm:itemgetter(*mm)(self.getCvPreds()))(),mm)
                       ) if not test else (ifelseLen(mm,zipl(self.names,self.models),
                                                        zipl(self.names[mm],self.models[mm])))
        if decVal:
            zizi= zipl(ifelseLen(mm,self.names,
                                self.names[mm]),
                    ifOneGetArr(ifelseLen(mm,lambda:self.getCvPreds(),
                                 lambda mm=mm:itemgetter(*mm)(self.getCvPreds()))(),mm),
                             ifOneGetArr(ifelseLen(mm,lambda:self.getCvDecFn(),
                                 lambda mm=mm:itemgetter(*mm)(self.getCvDecFn()))(),mm)
                   ) if not test else (ifelseLen(mm,zipl(self.names,self.models,None),
                                                    zipl(self.names[mm],self.models[mm],None)))
        return ifelse(enumerate_,lambda:enumerate(zizi),lambda:zizi)()
    
    def m_process_true(self,m,test=None):
        mm=flatArray(m)
        mm=self.getIndexFromNames(mm)
        test=ifelse(test,test,self.isModelsFitted())
        return ifelse(not test,(self.X_train,self.y_train),
                      (self.X_test,ifelseLen(mm,self.y_test,
                                self.y_test[mm])
                            ))
    
    def plot_classes_pred_erreurs(self,m=[],title=None,returnOK=F,rotation=0,
                                 show=True,figsize=(6,5),**xargs):
        _,y_true=self.m_process_true(m)
        #show=True
        #plt.figure(figsize=figsize)
        def beforePred(i):
            #i.predictions_=np.array(i.predictions_).T
            i.ax.xaxis.set_tick_params(rotation=rotation)
        o=[self.visualize_model(self.createFalseClassif(y_train,y_pred,index),
                              fit=False,
                              fn=studyClassif_viz.ClassPredictionError,
                              title=ifelse(title,title,name),show=show,
                                beforeShowFn=beforePred,name=name,**xargs)
                             for index,(name,y_pred) in  self.m_process_test(m)]
        #for i in o:
            #i.ax.xaxis.set_tick_params(rotation=0)
            #i.show()
        return ifelse(returnOK,o)
    
    def most_confused_global(self,m=[],lim=10,classNames=True,
                             returnOK=False,percent=False,roundVal=None,**xargs):
        _,y_true=self.m_process_true(m)
        res= {name+" (N°"+str(index)+")" :most_confused_global(y_true,y_pred,lim=lim,percent=percent)
                  for index,(name,y_pred) in self.m_process_test(m)}
        moreS=[j[1] for i,j in res.items()]
        res={i:caseConfusionGlobal.fromArr(j[0],self.namesCls,roundVal=roundVal) for i,j in res.items()}
        o=printDico(res,fn=printMostConfGlob,moreAdd=moreS)
        return ifelse(returnOK,o)
    
    def getObsConfused(self,classe,predit,m=[],lim=10,printOk=False,returnOK=True,shuffle=True):
        X_true,y_true=self.m_process_true(m)
        o=StudyDico({name:getObsMostConfused(classe,predit,X_true,y_true,y_pred,lim=lim) 
                     for index,(name,y_pred) in self.m_process_test(m)},
                    __study_i=classe,__study_j=predit,__study=self)
        if printOk: printDico(o,fn=printArr)
        return ifelse(returnOK,o)
    
    def addModelsToCurrentCV_(self,m=[]):
        mm=flatArray(m)
        self.setModels(np.array(self.models.tolist()+mm.tolist()))
    
    def addModelsToCurrentCV(self,m=[],id_=None,returnSelf=False):
        mm=flatArray(m)
        u=self.updateModels(mm)
        version=randomString()
        params=self.getCV()[2]
        params["cv"]=self.getCV()[0]
        u.computeCV(**params)
        res=merge(self.getCV()[1],u.getCV()[1])
        res=[self.getCV()[0],res,self.getCV()[2]]
        #self.history[version]=copy.deepcopy(self)
        self._cv[self._nameCV]=res
        self.addModelsToCurrentCV_(m)
        #self.setModels(np.array(self.models.tolist()+mm.tolist()))
        if returnSelf:
            return self
    
    @staticmethod
    def clone(self,id_=None,deep=False):
        id_=ifelse(id_,id_,self.id_)
        if deep:
            l=StudyClassif(id_=id_)
            l.__dict__ = merge_two_dicts(l.__dict__,self.__dict__.copy())
            l.id_=id_

            return l
        md=copy.deepcopy(self)
        md.id_=id_
        return md
    
    def clone(self,id_=None,deep=False):
        id_=ifelse(id_,id_,self.id_)
        if deep:
            l=StudyClassif(id_=id_)
            l.__dict__ = merge_two_dicts(l.__dict__.copy(),self.__dict__.copy())
            l.id_=id_
            return l
        md=copy.deepcopy(self)
        md.id_=id_
        return md
    
    def step_back_add(self):
        self=StudyDico(self.history)[-1]
        #return res
        
    def addModels(self,m=[]):
        mm=flatArray(m)
        inplace=False
        o=[self.models.tolist()+mm.tolist()]
        if inplace:
            self._init(o)
        else:
            return self.updateModels(o)
        
    def plot_classe_balance(self,test=False,both=False,show=True,returnOK=False,
                            figsize=(4,4),figsizeShow=(4,4),dpi=400,figure=None,getImgArgs={},showArgs={},**xargs):
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        cb=ClassBalance(labels=self._namesCls,ax=ax,fig=fig,**xargs)
        if not both: cb.fit(self.y_train if not test  else self.y_test)
        else: cb.fit(self.y_train,self.y_test)
        #
        img=IMG.getImg(yellow=True,visualizer=cb,**getImgArgs)
        if show:img.show(figsize=figsizeShow,dpi=dpi,figure=figure,**showArgs)
        return ifelse(returnOK,[cb,img])
    
    def plot_roc_auc(self,m=[],returnOK=False,figsize=(6,5),dpi=400,micro=F,size=(600, 360), macro=F,per_class=F,**xargs):
        test=False
        _,y_true=self.m_process_true(m)
        #plt.figure(figsize=figsize)
        #ot= [plt.subplots(figsize=figsize,dpi=dpi) for i in range(len(m))]
        def l(index,name,y_pred,decVal):
            fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
            return self.visualize_model(self.createFalseClassifDecFn(decVal,
                                                              y_true,y_pred,index),
                                 fit=False,size=size,
                                 fn=ROCAUC,micro=micro,fig=fig, macro=macro,per_class=per_class,name=name,figsize=figsize,**xargs)
        #decValues=self.getCvDecFn()
        o=[l(index,name,y_pred,decVal) for index,(name,y_pred,decVal) in self.m_process_test(m,decVal=T)]
        return ifelse(returnOK,o)
    
    def plot_precision_recall_curves(self,m=[],figsize=(6,5),returnOK=False,**xargs):
        test=False
        plt.figure(figsize=figsize)
        _,y_true=self.m_process_true(m)
        decValues=self.getCvDecFn()
        
        o=[
            self.visualize_model(self.createFalseClassifDecFn(decValues[index],
                                                              y_true,y_pred,index),
                                 fit=True,
                                 fn=studyClassif_viz.PrecisionRecallCurve,figsize=figsize,name=name,**xargs)
            for index,(name,y_pred) in self.m_process_test(m)]
        return ifelse(returnOK,o)
    
    
    
    def visualize_model(self,i=None,fit=True,fn=ClassificationReport,cmap="YlGn",show=True,size=(600, 360),title="",
                        noTest=True,fnFinalize=lambda *a,**ba:None,**xargs):
        ni=self._namesCls
        return visualize_model(self._X_train,self._y_train,
                               self._X_train if noTest else self.X_test,self._y_train if noTest else self.y_test,
                               i,fit,ni,fn,show=show,cmap=cmap,size=size,title=title,fnFinalize=fnFinalize,**xargs)
    def computeCV(self,cv=5,random_state=42,shuffle=True,classifier=True,
                 name=None,recreate=False,parallel=True,metric="accuracy",
                 models=None,noDiff=False):
        argu=dict(cv=cv,random_state=random_state,shuffle=shuffle,classifier=classifier,
            name=name,recreate=recreate,parallel=parallel,metric=metric,models=models)
        models=ifelse(models,models,self.models)
        #if cv == 1 :
        #    crossV.crossV(self.models, self.X_train, self.y_train ,train, test,self.X_test,self.y_test)
        cv=computeCV(X=self._X_train,y=self._y_train,cv=cv,classifier=classifier,random_state=random_state,
                     shuffle=shuffle)
        if name is None:
            self._nameCV = "CV_"+randomString(10) 
            cvo=self.__crossV(cv,parallel=parallel,metric=metric,models=models)
            self._cv[self._nameCV]=[cv,cvo,argu]
        else:
            #delattr(self,"_nameCV")
            if (name in self._cv and recreate) or (name not in self._cv):
                cvo=self.__crossV(cv,parallel=parallel,metric=metric,models=models)
                self._cv[name]=[cv,cvo,argu]
                self._nameCV=name
            else:
                nn=newStringUniqueInDico(name,self._cv)
                a=input("[computeCV] name '{}' is already take recreate (y/N)?".format(name))
                if a=="y":
                    self.computeCV(cv=cv,random_state=random_state,shuffle=shuffle,classifier=classifier,
                                  name=name,recreate=True,parallel=parallel,metric=metric)
                    return
                with ShowWarningsTmp():
                    warnings.warn("[computeCV] name '{}' is already take, mtn c'est '{}'".format(name,nn))
                self.computeCV(cv=cv,random_state=random_state,shuffle=shuffle,classifier=classifier,
                                  name=nn,recreate=recreate,parallel=parallel,metric=metric,models=models)
        if not noDiff:
            _=self._diff[0]
                           
    def __crossV(self,cv=3,verbose=0,n_jobs=-1,parallel=True,metric="accuracy",models=None):
        cvv=cv
        scorer=get_metric(metric)
        #models=ifelse(models,models,self.models)
                           
        cvvTr=[i[0] for i in cvv]
        cvvVal=[i[1] for i in cvv]
                           
        cvvTrCon=np.argsort(np.concatenate(cvvTr))
        cvvValCon=np.argsort(np.concatenate(cvvVal))
        
        resu2=[cross_validate(mod ,self.X_train,self.y_train,return_train_score=True,
                            return_estimator=True,cv=cvv,n_jobs=ifelse(parallel,n_jobs),verbose=verbose,scoring=metric) for mod in models]

        preduVal=[[i.predict(self.X_train[k]) for i,k in zipl(resuI["estimator"],cvvVal) ] for resuI in resu2]
                           
        preduuVal=[np.concatenate(preduI)[cvvValCon] for preduI in preduVal]
        
        scoreVal = [resuI["test_score"] for resuI in resu2]
        
        preduTr=[[i.predict(self.X_train[k]) for i,k in zipl(resuI["estimator"],cvvTr) ] for resuI in resu2]
                           
        preduuTr=[np.concatenate(preduI)[cvvTrCon] for preduI in preduTr]
        
        scoreTr = [resuI["train_score"] for resuI in resu2]
        
        decVal=[[getDecFn(i,self.X_train[k]) for i,k in zipl(resuI["estimator"],cvvVal) ] for resuI in resu2]
        decVal2=[concatenateDecFn(preduI,cvvValCon) for preduI in decVal]
        return { "preds": {
                            "Tr":{"original":preduTr,"sorted":preduuTr},
                            "Val":{"original":preduVal,"sorted":preduuVal}
                            },
                "scores": {
                            "Tr":scoreTr,
                            "Val":scoreVal
                            },
                "cv":{ 
                        "splitted":cvv,
                        "Tr":{"original":cvvTr,"argsort":cvvTrCon},
                        "Val":{"original":cvvVal,"argsort":cvvValCon},
                        "cv_validate":resu2
                        },
                "decFn":{
                    "Val":{"original":decVal,"sorted":decVal2}
                }
        }
                           
  
    def __getattr__(self,a):
        if has_method(self,"_"+a): return getattr(self,"_"+a,None)
        else: raise AttributeError(a)
            
    def getCV(self,name=None):
        if name is None:
            name=self._nameCV
        return self._cv[name]
    
    def getCvEstimator(self,n=0):
        return self.getCV()[1]["cv"]["cv_validate"][0]["estimator"][n]
    
    def getCvPreds(self,type_="Val",isSorted=True):
        r=self.getCV()[1]["preds"][type_]
        return r["sorted"] if isSorted else r["original"]
    
    def getCvDecFn(self,type_="Val",isSorted=True):
        r=self.getCV()[1]["decFn"][type_]
        return r["sorted"] if isSorted else r["original"]
    
    def createFalseClassif(self,y_true,y_pred,n=0):
        return FalseClassif(y_true,y_pred,self.getCvEstimator(n=n))
    
    def createFalseClassifDecFn(self,decVal,y_true,y_pred,n=0):
        
        return FalseClassifDecFn(decVal,y_true,y_pred,self.getCvEstimator(n=n))
    
    def cvFromOtherStudy(self,other):
        self._cv=other._cv
        self._nameCV = other._nameCV
    def getObsNotErrors(vt,i,j):
        i=vt.getIndexFromNames([i])
        j=vt.getIndexFromNames([j])
        return np.where((np.logical_not(vt.diff[3][i[0]])| np.logical_not( vt.diff[3][j[0]])))[0]
    
    def getObsErrors(vt,i,j):
        i=vt.getIndexFromNames([i])
        j=vt.getIndexFromNames([j])
        return np.where((vt.diff[3][i[0]]& vt.diff[3][j[0]]))[0]
class studyClassif_Img:
    @staticmethod
    def plotImgsMultiClassif(im,title="",nr=2,nc=5,figsize=(9,5),w=28,h=28,titleSize=29,m=[],reshape=False):
        mm=flatArray(m)
        mm=getattr(im,"__study").getIndexFromNames(mm)
        names=getattr(im,"__study").names
        names = names if len(mm)==0 else names[mm]
        uu=studyClassif_Img.reshapeMultiClassif(im,w,h)  if reshape else im
        uu = uu if len(mm) ==0 else np.array(uu)[mm]
        for i in range(len(uu)):
            studyClassif_Img.plotImgs(im[i],names[i]+"\n",nr,nc,figsize,w,h,titleSize,not reshape,addToTitle=T)
        
    @staticmethod
    def plotImgs(im,title="",nr=2,nc=5,figsize=(9,5),w=28,h=28,titleSize=29,reshape=True,addToTitle=False):
        title_ = "Classe {} prédit en {}".format(getattr(im,"__study_i"),getattr(im,"__study_j"))
        if (len(title)==0) or (len(title) >0 and addToTitle): title=title+title_ 
        studyClassif_Img._plotImgs(im,title,nr,nc,w,h,w,h,figsize,titleSize,reshape)
                           
    @staticmethod
    def plotDigits(im,title="",elemsByRows=10,figsize=(9,5),w=28,h=28,titleSize=29,reshape=True,addToTitle=False,lim=10):
        title_ = "Classe {} prédit en {}".format(getattr(im,"__study_i"),getattr(im,"__study_j"))
        if (len(title)==0) or (len(title) >0 and addToTitle): title=title+title_ 
        studyClassif_Img._plotImgs(im,title,elemsByRows,w,h,figsize,titleSize,reshape,lim=lim)
                           
    @staticmethod
    def _plotImgs(im,title="",nr=2,nc=5,figsize=(9,5),w=28,h=28,titleSize=29,reshape=True,*args,**xargs):
        uu=studyClassif_Img.reshape(im,w,h) if reshape else im
        plt.figure(figsize=figsize)
        for _i,i in enumerate(uu): 
            plt.subplot(nr,nc,_i+1)
            plt.imshow(i)
            plt.axis('off')
        plt.suptitle(title,size=titleSize);
    @staticmethod
    def reshapeMultiClassif(im,w=28,h=28):
        return [studyClassif_Img.reshape(j,w,h) for ww,j in im.items()]
    
    @staticmethod
    def reshape(im,w=28,h=28):
        return [i.reshape(w,h) for i in im]
    
    @staticmethod
    def _plotDigits(instances,title="", elemsByRows=10,w=28,h=28,figsize=(9,5),
                    titleSize=29,reshape=True,lim=10,show=True,returnOK=False,noImg=False,ax=None, **options):
        instances=instances[:lim]
        images_per_row=elemsByRows
        images_per_row = min(len(instances), images_per_row)
        images = instances if not reshape else [instance.reshape(w,h) for instance in instances]
        n_rows = (len(instances) - 1) // images_per_row + 1
        row_images = []
        n_empty = n_rows * images_per_row - len(instances)
        images.append(np.zeros((w, h * n_empty)))
        for row in range(n_rows):
            rimages = images[row * images_per_row : (row + 1) * images_per_row]
            row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
        if ax is None:
            plt.figure(figsize=figsize)
        if ax is None:
            plt.imshow(image, cmap = mpl.cm.binary, **options)
        else:
            ax.imshow(image, cmap = mpl.cm.binary, **options)
        if ax is None:
            plt.title(title,size=titleSize)
            plt.axis("off")
        else:
            ax.set_title(title,size=titleSize)
            ax.axis("off")
        if not noImg:
            img=IMG.getImg()
            if show:img.show(figsize=figsize)
            return ifelse(returnOK,img)


def __partial(*args,**xargs):
    return lambda x: pipe_partial(x,[args,xargs])
def concatenateDecFn(ll,k):
    #print(np.ndim(ll))
    #print(np.ndim(ll[0]))
    if np.ndim(ll[0])==3:
        dfr=np.concatenate([i for i in ll],axis=1)
        dfr=[i[k] for i  in dfr]
        return dfr
    return np.concatenate(ll)[k]
def __c(*args,**xargs):
    return [args,xargs]
import builtins
class fakeSelfo(object):
    def __init__(self,globally=True):
        self.globally=globally
    def __getattr__(self,a):
        if a in globals():
            return globals()[a]
        if has_method(builtins,a):
            return getattr(builtins,a)
        return None
        #return ifelse(self.globally,lambda:globals()[a],lambda: ifelse() fnRien)()
    
class StudyGlobal:
    def __init__(self,name):
        self.name=name
        self.studies={}
        self.curr=None
        self.data={"_ZERO":[None,None,None,None,None]}
        
    def add(self,study_):
        study_2=study_
        study_2.fromGlobal=self
        self.studies[study_.id_]=study_2
        self.curr=study_2.id_
        return study_2

    
    def get(self,id_=None):
        id_=ifelse(id_,id_,self.curr)
        return self.studies[id_]
    
    def getAndCurr(self,id_):
        self.curr=id_
        return self.get(od_)
        
    def addOrGetStudy(self,id_,cls,recreate=False,clone=False,deep=True):
        def clonee(rrt):
            return getStaticMethodFromObj(rrt,"clone")(rrt,deep=deep)
        def recreatee():
            rrt=cls(id_=id_)
            if clone :
                rrt=clonee(rrt)
            res =self.add(rrt)
            return res
        def cloneStudy():
            ru=self.studies[id_] 
            ru=clonee(ru)
            self.studies[id_]=ru
            return self.studies[id_]
        if recreate:
            res=recreatee()
        else:
            res= ifelse(id_ in self.studies,
                      lambda:self.studies[id_] if not clone else cloneStudy() ,
                      lambda:recreatee())()
        if not res.isProcess and res.processingDataFn is not None:
            warnings.warn("Attention vous devez appeler impérativement  la méthode preprocessingData de l'object 'StudyClassif' reçu pour que les données soit les bonnes") 
        self.curr=id_
        return res
    
    @property
    def getCurr(self):
        return self.get()
        
    
    @staticmethod
    def getOrCreate(name,repertoire="study_globals",ext=".studyGlobal",
                    path=os.getcwd(),delim="/",recreate=F,dontDoPreprocess=False,clone=False,deep=True,**xargs):
        def clonee(rrt):
            return getStaticMethodFromObj(rrt,"clone")(rrt,deep=deep)
        repo=path+delim+repertoire
        if recreate:
            return StudyGlobal(name)
        if not os.path.exists(repo):
            return StudyGlobal(name)
        filo=repo+delim+name+ext
        if not os.path.isfile(filo):
            return StudyGlobal(name)
        sl=SaveLoad.load(filo,**xargs)
        sf={}
        for k,v_ in sl.studies.items():
            #print(k)
            v=ifelse(clone,lambda: clonee(v_),lambda:v_)()
            if v.fromGlobal is not None and v.fromGlobal== True and v.idData is not None:
                v.fromGlobal=sl
                #print(v.idData)
                v.setDataTrainTest(id_=v.idData)
                if not dontDoPreprocess: 
                    try:
                        v.preprocessingData(v.processingDataFn)
                        #print(v.isProcess)
                    except Exception as inst:
                        if v.processingDataFn is not None: v.isProcess=False
                    #print("Error")
                    #print(inst)
                        #print(v.isProcess)
                v.finalCheck()
            sf[k]=v
        sl.studies=sf
        return sl
    
    def save(self,repertoire="study_globals",ext=".studyGlobal",path=os.getcwd(),
             delim="/",returnOK=False,noDiff=False,**xargs):
        name=self.name
        repo=path+delim+repertoire
        if not os.path.exists(repo):
            os.makedirs(repo)
        filo=repo+delim+name+ext
        sl=StudyGlobal.clone(self,deep=True)
        ff={}
        for k,v in sl.studies.items():
            #print(k,v.fromGlobal is not None,v.idData is not None)
            if v.fromGlobal is not None and v.idData is not None:
                li=v.idData
                if noDiff:
                    v._diff=None
                #print(k,not noDiff,type(v._diff),loadFnIfCallOrSubset.mine(v._diff))
                if not noDiff and not loadFnIfCallOrSubset.mine(v._diff):
                    if getClassName(v._diff) is not None and getClassName(v._diff)=="loadFnIfCallOrSubset":
                        v._diff=None
                        v.finalCheck()
                if not noDiff and loadFnIfCallOrSubset.mine(v._diff):
                    try:
                        v._diff.load(v)
                    except:
                        #raise Exception("pb")
                        v._diff=None
                        warnings.warn("diff de {} ne peux etre load".format(k))
                v.setDataTrainTest(id_="_ZERO")
                v.idData=li
                v.fromGlobal=True
                ff[k]=v
        #return sl
        sl.studies=ff
        if returnOK:
            return sl
        else:
            SaveLoad.save(sl,filo,**xargs)
    
    
    
    def clone_(self,name=None,deep=False):
        name = ifelse(name,name,self.name)
        #print( name)
        if deep:
            l=.__class__(name)
            l.__dict__ = merge_two_dicts(l.__dict__.copy(),self.__dict__.copy())
            l.name=name
            #print("ici")
            l.studies={k:v.clone(v.id_,deep=True) for (k,v) in self.studies.items()}
            return l
        else:
            me=copy.deepcopy(self)
            me.name=name
            return me
        
    @staticmethod
    def clone(self,ID=None,deep=False):
        name = ifelse(name,name,self.name)
        #print( name)
        if deep:
            l=self.__class__(ID)
            l.__dict__ = merge_two_dicts(l.__dict__,self.__dict__.copy())
            #print(l.__dict__)
            l.name=name
            l.studies={k:getStaticMethodFromObj(v,"clone")(v,v.id_,deep=True) for k,v in self.studies.items()}
            #print(l.__dict__)
            return l
        else:
            me=copy.deepcopy(self)
            me.name=name
            return me
    
    @staticmethod
    def fromStudyGlobal(self,studyG):
        return studyG.clone(studyG.name)
    
    def saveDatasWithId(self,id_,X_train,y_train,X_test,y_test,names=None):
        #id_=ifelse(id_,id_,randomString())
        self.data[id_]=[X_train,X_test,y_train,y_test,names]
        
from inspect import getsource
getsourceP = lambda a:print(getsource(a))
patch_all2()

def getStaticMethodFromObj(obj,met):
    mod=obj.__module__
    cls=obj.__class__.__name__
    return getStaticMethod(mod,cls,met)

def getStaticMethod(mod,cls,met):        
    modu=__import__(mod)
    clss=getattr(modu,cls)
    met=getattr(clss,met)
    return met
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z
import os
import sys
import warnings
import dill 
from types import ClassType
dill._dill._reverse_typemap['ClassType'] = ClassType




_DEFAULT_EXTENSION_MAP = {
    None: ".pkl",
    "pickle": ".pkl",
    "gzip": ".gz",
    "bz2": ".bz",
    "lzma": ".lzma",
    "zipfile": ".zip",
}

_DEFAULT_COMPRESSION_WRITE_MODES = {
    None: r"wb+",
    "pickle": r"wb+",
    "gzip": r"wb",
    "bz2": r"wb",
    "lzma": r"wb",
    "zipfile": r"w",
}

_DEFAULT_COMPRESSION_READ_MODES = {
    None: r"rb+",
    "pickle": r"rb+",
    "gzip": r"rb",
    "bz2": r"rb",
    "lzma": r"rb",
    "zipfile": r"r",
}


def get_known_compressions():
    """Get a list of known compression protocols
    Returns
    -------
    compressions: list
        List of known compression protocol names.
    """
    return [c for c in _DEFAULT_EXTENSION_MAP]


def get_default_compression_mapping():
    """Get a mapping from known compression protocols to the default filename
    extensions.
    Returns
    -------
    compression_map: dict
        Dictionary that maps known compression protocol names to their default
        file extension.
    """
    return _DEFAULT_EXTENSION_MAP.copy()


def get_compression_write_mode(compression):
    """Get the compression's default mode for openning the file buffer for
    writing.
    Returns
    -------
    write_mode_map: dict
        Dictionary that maps known compression protocol names to default write
        mode used to open files for
        :func:`~compress_pickle.compress_pickle.dump`.
    """
    try:
        return _DEFAULT_COMPRESSION_WRITE_MODES[compression]
    except Exception:
        raise ValueError(
            "Unknown compression {}. Available values are: {}".format(
                compression, list(_DEFAULT_COMPRESSION_WRITE_MODES.keys())
            )
        )


def get_compression_read_mode(compression):
    """Get the compression's default mode for openning the file buffer for
    reading.
    Returns
    -------
    read_mode_map: dict
        Dictionary that maps known compression protocol names to default write
        mode used to open files for
        :func:`~compress_pickle.compress_pickle.load`.
    """
    try:
        return _DEFAULT_COMPRESSION_READ_MODES[compression]
    except Exception:
        raise ValueError(
            "Unknown compression {}. Available values are: {}".format(
                compression, list(_DEFAULT_COMPRESSION_READ_MODES.keys())
            )
        )


def set_default_extensions(filename, compression=None):
    """Set the filename's extension to the default that corresponds to
    a given compression protocol. If the filename already has a known extension
    (a default extension of a known compression protocol) it is removed
    beforehand.
    Parameters
    ----------
    filename: str
        The filename to which to set the default extension
    compression: None or str (optional)
        A compression protocol. To see the known compression protocolos, use
        :func:`~compress_pickle.compress_pickle.get_known_compressions`
    Returns
    -------
    filename: str
        The filename with the extension set to the default given by the
        compression protocol.
    Notes
    -----
    To see the mapping between known compression protocols and filename
    extensions, call the function
    :func:`~compress_pickle.compress_pickle.get_default_compression_mapping`.
    """
    default_extension = _DEFAULT_EXTENSION_MAP[compression]
    if not filename.endswith(default_extension):
        for ext in _DEFAULT_EXTENSION_MAP.values():
            if ext == default_extension:
                continue
            if filename.endswith(ext):
                filename = filename[: (len(filename) - len(ext))]
                break
        filename += default_extension
    return filename


def infer_compression_from_filename(filename, unhandled_extensions="raise"):
    """Infer the compression protocol by the filename's extension. This
    looks-up the default compression to extension mapping given by
    :func:`~compress_pickle.compress_pickle.get_default_compression_mapping`.
    Parameters
    ----------
    filename: str
        The filename for which to infer the compression protocol
    unhandled_extensions: str (optional)
        Specify what to do if the extension is not understood. Can be
        "ignore" (do nothing), "warn" (issue warning) or "raise" (raise a
        ValueError).
    Returns
    -------
    compression: str
        The inferred compression protocol's string
    Notes
    -----
    To see the mapping between known compression protocols and filename
    extensions, call the function
    :func:`~compress_pickle.compress_pickle.get_default_compression_mapping`.
    """
    if unhandled_extensions not in ["ignore", "warn", "raise"]:
        raise ValueError(
            "Unknown 'unhandled_extensions' value {}. Allowed values are "
            "'ignore', 'warn' or 'raise'".format(unhandled_extensions)
        )
    extension = os.path.splitext(filename)[1]
    compression = None
    for comp, ext in _DEFAULT_EXTENSION_MAP.items():
        if comp is None:
            continue
        if ext == extension:
            compression = comp
            break
    if compression is None and extension != ".pkl":
        if unhandled_extensions == "raise":
            raise ValueError(
                "Cannot infer compression protocol from filename {} "
                "with extension {}".format(filename, extension)
            )
        elif unhandled_extensions == "warn":
            warnings.warn(
                "Cannot infer compression protocol from filename {} "
                "with extension {}".format(filename, extension),
                category=RuntimeWarning,
            )
    return compression


def compress_pickle_dump(
    obj,
    path,
    compression="infer",
    mode=None,
    protocol=-1,
    fix_imports=True,
    unhandled_extensions="raise",
    set_default_extension=True,
    **kwargs
):
    r"""Dump the contents of an object to disk, to the supplied path, using a
    given compression protocol.
    For example, if ``gzip`` compression is specified, the file buffer is
    opened as ``gzip.open`` and the desired content is dumped into the buffer
    using a normal ``pickle.dump`` call.
    Parameters
    ----------
    obj: any
        The object that will be saved to disk
    path: str
        The path to the file to which to dump ``obj``
    compression: None or str (optional)
        The compression protocol to use. By default, the compression is
        inferred from the path's extension. To see available compression
        protocols refer to
        :func:`~compress_pickle.compress_pickle.get_known_compressions`.
    mode: None or str (optional)
        Mode with which to open the file buffer. The default changes according
        to the compression protocol. Refer to
        :func:`~compress_pickle.compress_pickle.get_compression_write_mode` to
        see the defaults.
    protocol: int (optional)
        Pickle protocol to use
    fix_imports: bool (optional)
        If ``fix_imports`` is ``True`` and ``protocol`` is less than 3, pickle
        will try to map the new Python 3 names to the old module names used
        in Python 2, so that the pickle data stream is readable with Python 2.
    set_default_extension: bool (optional)
        If ``True``, the default extension given the provided compression
        protocol is set to the supplied ``path``. Refer to
        :func:`~compress_pickle.compress_pickle.set_default_extensions` for
        more information.
    unhandled_extensions: str (optional)
        Specify what to do if the extension is not understood when inferring
        the compression protocol from the provided path. Can be "ignore" (use
        ".pkl"), "warn" (issue warning and use ".pkl") or "raise" (raise a
        ValueError).
    kwargs:
        Any extra keyword arguments are passed to the compressed file opening
        protocol.
    Notes
    -----
    To see the mapping between known compression protocols and filename
    extensions, call the function
    :func:`~compress_pickle.compress_pickle.get_default_compression_mapping`.
    """
    if compression == "infer":
        compression = infer_compression_from_filename(path, unhandled_extensions)
    if set_default_extension:
        path = set_default_extensions(path, compression=compression)
    arch = None
    if mode is None:
        mode = get_compression_write_mode(compression)
    if compression is None or compression == "pickle":
        file = open(path, mode=mode)
    elif compression == "gzip":
        import gzip

        file = gzip.open(path, mode=mode, **kwargs)
    elif compression == "bz2":
        import bz2

        file = bz2.open(path, mode=mode, **kwargs)
    elif compression == "lzma":
        import lzma

        file = lzma.open(path, mode=mode, **kwargs)
    elif compression == "zipfile":
        import zipfile

        arch = zipfile.ZipFile(path, mode=mode, **kwargs)
        if sys.version_info < (3, 6):
            arcname = os.path.basename(path)
            arch.write(path, arcname=arcname)
        else:
            file = arch.open(path, mode=mode)
    if arch is not None:
        with arch:
            if sys.version_info < (3, 6):
                buff = dill.dumps(obj, protocol=protocol)
                arch.writestr(arcname, buff)
            else:
                with file:
                    dill.dump(obj, file, protocol=protocol)
    else:
        with file:
            dill.dump(obj, file, protocol=protocol)


def compress_pickle_load(
    path,
    compression="infer",
    mode=None,
    fix_imports=True,
    encoding="ASCII",
    errors="strict",
    set_default_extension=True,
    unhandled_extensions="raise",
    **kwargs
):
    r"""Load an object from a file stored in disk, given compression protocol.
    For example, if ``gzip`` compression is specified, the file buffer is opened
    as ``gzip.open`` and the desired content is loaded from the open buffer
    using a normal ``pickle.load`` call.
    Parameters
    ----------
    path: str
        The path to the file from which to load the ``obj``
    compression: None or str (optional)
        The compression protocol to use. By default, the compression is
        inferred from the path's extension. To see available compression
        protocols refer to
        :func:`~compress_pickle.compress_pickle.get_known_compressions`.
    mode: None or str (optional)
        Mode with which to open the file buffer. The default changes according
        to the compression protocol. Refer to
        :func:`~compress_pickle.compress_pickle.get_compression_read_mode` to
        see the defaults.
    fix_imports: bool (optional)
        If ``fix_imports`` is ``True`` and ``protocol`` is less than 3, pickle
        will try to map the new Python 3 names to the old module names used
        in Python 2, so that the pickle data stream is readable with Python 2.
    encoding: str (optional)
        Tells pickle how to decode 8-bit string instances pickled by Python 2.
        Refer to the standard ``pickle`` documentation for details.
    errors: str (optional)
        Tells pickle how to decode 8-bit string instances pickled by Python 2.
        Refer to the standard ``pickle`` documentation for details.
    set_default_extension: bool (optional)
        If `True`, the default extension given the provided compression
        protocol is set to the supplied `path`. Refer to
        :func:`~compress_pickle.compress_pickle.set_default_extensions` for
        more information.
    unhandled_extensions: str (optional)
        Specify what to do if the extension is not understood when inferring
        the compression protocol from the provided path. Can be "ignore" (use
        ".pkl"), "warn" (issue warning and use ".pkl") or "raise" (raise a
        ValueError).
    kwargs:
        Any extra keyword arguments are passed to the compressed file opening
        protocol.
    Returns
    -------
    The unpickled object: any
    Notes
    -----
    To see the mapping between known compression protocols and filename
    extensions, call the function
    :func:`~compress_pickle.compress_pickle.get_default_compression_mapping`.
    """
    if compression == "infer":
        compression = infer_compression_from_filename(path, unhandled_extensions)
    if set_default_extension:
        path = set_default_extensions(path, compression=compression)
    if mode is None:
        mode = get_compression_read_mode(compression)
    arch = None
    if compression is None or compression == "pickle":
        file = open(path, mode=mode)
    elif compression == "gzip":
        import gzip

        file = gzip.open(path, mode=mode, **kwargs)
    elif compression == "bz2":
        import bz2

        file = bz2.open(path, mode=mode, **kwargs)
    elif compression == "lzma":
        import lzma

        file = lzma.open(path, mode=mode, **kwargs)
    elif compression == "zipfile":
        import zipfile

        arch = zipfile.ZipFile(path, mode=mode, **kwargs)
        file = arch.open(path, mode=mode)
    if arch is not None:
        with arch:
            with file:
                output = dill.load(
                    file
                )
    else:
        with file:
            output = dill.load(
                file
            )
    return output
compress_pickle=StudyDico(dict(load=compress_pickle_load,dump=compress_pickle_dump))