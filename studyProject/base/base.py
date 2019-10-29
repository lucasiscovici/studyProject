from ..utils import studyDico, isStr, get_metric, ifelse, randomString, uniquify, \
                    getClassName, rangel, SaveLoad, merge_two_dicts, \
                    newStringUniqueInDico, check_cv2, Obj, mapl, ifNotNone, has_method ,\
                    isInt, zipl, BeautifulDico,BeautifulList, getStaticMethodFromObj,\
                    takeInObjIfInArr, convertCamelToSnake, getAnnotationInit, securerRepr, merge_dicts
from sklearn.metrics import make_scorer, get_scorer
from sklearn.model_selection import cross_validate
import numpy as np
import copy
import os
import warnings
from interface import implements, Interface
from abc import ABCMeta, abstractmethod, ABC
# from typing import get_origin
# class ImportExportLoadSaveClone(Interface):
try:
    from typing import TypeVar, _GenericAlias
except:
    from typing import GenericMeta as _GenericAlias
    class TypeVar:
        def __init__(self,*args,**xargs):pass
import sys
import inspect

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
class BaseSupFactory:

    def __init__(self):
        self._classes = {}

    def register_class(self, class_):
        self._classes[class_.__name__] = class_

    def get_class(self, class_name):
        class_ = self._classes.get(class_name)
        if not class_:
            raise ValueError(class_name)
        return class_
factoryCls= BaseSupFactory()

def str2Class(str):
    try:
        rep=getattr(sys.modules[__name__], str)
    except Exception as e2:
        try:
            rep=factoryCls.get_class(str)
        except Exception as e:
            raise e
    return rep
def get_origin(l):
    try:
        rep=l.__orig_bases__[0]
    except:
        try:
            rep=l.__origin__
        except:
            rep=l
    return rep
def get_args(l):
    try:
        rep=l.__args__
    except:
        rep=l
    if rep is None:
        rep=[TypeVar("blabla")]
    return rep
class Base:
    DEFAULT_PATH="__studyFiles"
    DEFAULT_REP="study_"
    DEFAULT_EXT=".study"
    EXPORTABLE=["ID"]
    EXPORTABLE_SUFFIX="EXP"
    EXPORTABLE_ARGS={}

    @classmethod
    def get_repertoire(cls,repertoire,suffix=None,chut=True):
        suffix=cls.EXPORTABLE_SUFFIX if suffix is None else suffix
        if repertoire is None:
            if cls.__name__ != Base.__name__ and cls.DEFAULT_REP!= Base.DEFAULT_REP:
                repertoire=cls.DEFAULT_REP+suffix
            else:
                repertoire=cls.DEFAULT_REP+convertCamelToSnake(cls.__name__)+suffix
                if not chut:
                    warnings.warn("\n[Base save] repertoire est non spécifié, repertoire = {} ".format(repertoire))
        return repertoire

    @classmethod
    def get_ext(cls,ext,suffix=None,chut=True):
        suffix=cls.EXPORTABLE_SUFFIX if suffix is None else suffix
        if ext is None:
            if cls.__name__ != Base.__name__ and cls.DEFAULT_REP!= Base.DEFAULT_REP:
                ext =cls.DEFAULT_EXT+suffix
            else:
                ext=cls.DEFAULT_EXT+convertCamelToSnake(cls.__name__)+suffix
                if not chut:
                    warnings.warn("\n[Base save] ext est non spécifié, ext = {} ".format(ext))
        return ext

    def __init__(self,ID:str=None):
        self.ID=ifelse(ID is None,lambda:randomString(),lambda:ID)() 

    def clone(self,name=None,deep=True):
        return self.__class__.Clone(self,name,deep)
        
    @staticmethod
    def Clone(self,ID=None,deep=True):
        ID = ifelse(ID is not None,ID,self.ID)
        #print( name)
        if deep:
            exported=self.export(save=False)
            l=self.__class__.import__(self.__class__(),exported)
            l.ID=ID

            # l=self.__class__(ID)
            # l.__dict__ = merge_two_dicts(l.__dict__,self.__dict__.copy())
            # #print(l.__dict__)
            # l.ID=ID
            #print(l.__dict__)
            return l
        else:
            me=copy.deepcopy(self)
            me.ID=ID
            return me

    @classmethod
    def build_repertoire(cls,repertoire,path=os.getcwd(),dp=None,delim="/",
        fn=lambda repo:os.makedirs(repo)):
        dp=cls.DEFAULT_PATH if dp is None else dp 
        repo=delim.join([i for i in [path,dp,repertoire] if i != ""])
        if not os.path.exists(repo):
            fn(repo)
        return repo

    @classmethod 
    def build_ext(cls,repo,ext,ID,delim="/",recreate=False,chut=True):
        ok=True
        filo=repo+delim+ID+ext
        if os.path.isfile(filo) and not recreate:
            ok=False
            if not chut:
                warnings.warn("\n[Base save] {} exite deja est recreate est à faux".format(filo))
        return (ok,filo)

    @classmethod
    def get_rep_ext(cls,repo,ext,chut=True):
        return (cls.get_repertoire(repo,chut=chut),
                cls.get_ext(ext,chut=chut))

    @classmethod
    def build_rep_ext(cls,repertoire,ext,ID,path=os.getcwd(),dp=None,delim="/",
                        fn=lambda repo:os.makedirs(repo),recreate=False,chut=True):
        repo=cls.build_repertoire(repertoire,path=path,dp=dp,delim=delim,fn=fn)
        return (repo,
            cls.build_ext(repo,ext,ID,delim,recreate,chut))
    @classmethod
    def Save(cls,self,
             ID,
             repertoire=None,
             ext=None,
             path=os.getcwd(),
             delim="/",
             recreate=False,
             suffix="",
             chut=True,
             noDefaults=False,
             **xargs):
        ID=self.ID if ID is None else ID
        if noDefaults:
            repertoire=""
            ext=""


        repertoire,ext=cls.get_rep_ext(repertoire,ext,chut=chut)

        repo,(ok,filo)=cls.build_rep_ext(repertoire,ext,ID,dp=cls.DEFAULT_PATH if not noDefaults else "",
                                            chut=chut,recreate=recreate)
        # print(ok)
        if ok:
            SaveLoad.save(self,filo,chut=chut,**xargs)
    
    def save(self,
             repertoire=None,
             ext=None,
             ID=None,
             path=os.getcwd(),
             delim="/",
             recreate=False,
             **xargs):
        self.__class__.Save(self,ID,repertoire,ext,path,delim,recreate,**xargs)
    
    @classmethod
    def Load(cls,ID,
             repertoire=None,
             ext=None,
             path=os.getcwd(),
             delim="/",
             suffix="",
             chut=True,
             noDefaults=False,
            **xargs):
        if noDefaults:
            repertoire=""
        if repertoire is None:
            if cls.__name__ != Base.__name__ and cls.DEFAULT_REP!= Base.DEFAULT_REP:
                repertoire =cls.DEFAULT_REP+suffix
            else:
                repertoire=cls.DEFAULT_REP+convertCamelToSnake(cls.__name__)+suffix
                if not chut:
                    warnings.warn("\n[Base load] repertoire est non spécifié, repertoire = {} ".format(repertoire)) 
        if noDefaults:
            ext=""
        if ext is None:
            if cls.__name__ != Base.__name__ and cls.DEFAULT_REP!= Base.DEFAULT_REP:
                ext =cls.DEFAULT_EXT+suffix
            else:
                ext=cls.DEFAULT_EXT+convertCamelToSnake(cls.__name__)+suffix
                if not chut:
                    warnings.warn("\n[Base load] ext est non spécifié, ext = {} ".format(ext))
        dp=cls.DEFAULT_PATH if not noDefaults else "" 
        repo=delim.join([i for i in [path,dp,repertoire] if i != ""])
        filo=repo+delim+ID+ext
        if not os.path.isfile(filo):
            if not chut:
                warnings.warn("\n[Base load] {} n'exite pas ".format(filo))
        try:
            resu=SaveLoad.load(filo,**xargs)
        except Exception as e:
            raise e
            resu=None
        return resu

 
    @staticmethod
    def import___(cls,ol,loaded):
        if loaded is None:
            return None
        if isinstance(loaded,dict) and len(loaded)==0:
            return {}
        if isinstance(loaded,list) and len(loaded)==0:
            return []
        if isinstance(loaded,tuple) and len(loaded)==0:
            return tuple()
        # argus=get_default_args(cls.import__)
        for i in cls.__bases__:
            # print(i)
            if hasattr(i,"import__"):
                # if "me" in argus:
                #     if argus["me"] == i.__name__:
                #         continue
                # print('ok')
                # if cls.__name__ == "BaseSuperviseProject":
                #     print("")
                # print("hi",i)
                ol=i.import__(ol,loaded)
        # try:
        #     rrr=cls.__base__.__import
        #     # rep=getAnnotationInit(rep)
        # except:
        #     rrr=None
        #     pass
        # if rrr is not None:
        #     ol=rrr(ol,loaded)
        # print(cls.__name__)
        # print(ol)
        f=cls.EXPORTABLE
        ff=cls.EXPORTABLE_ARGS
        annot=getAnnotationInit(cls())
        if "underscore" in ff:
            annot={("_"+k):v for k,v in annot.items() if k in f}
        else:
            annot={k:v for k,v in annot.items() if k in f}
        if "underscore" in ff:
            annot2=["_"+i for i in f if "_"+i not in annot]
        else:
            annot2=[i for i in f if i not in annot]
        for k,v in annot.items():
            # print(k)
            # print(cls.__name__)
            if get_origin(v) is list:
                cl=get_args(v)[0] if isinstance(v,_GenericAlias) else v
                if isinstance(cl,TypeVar) or not isinstance(cl(),Base):
                    repo=loaded[k]
                else:
                    kk=loaded[k][0] if len(loaded[k])>0 else None
                    cl2=str2Class(kk["____cls"]) if kk is not None and "____cls" in kk else cl
                    repo=[ cl2.import__(cl2(),i) for i in loaded[k] ]
                # setattr(ol,k,repo)
            elif get_origin(v) is dict:
                cl=get_args(v)[1] if isinstance(v,_GenericAlias) else v
                if isinstance(cl,TypeVar) or not isinstance(cl(),Base):
                    repo=loaded[k]
                else:
                    kk=list(loaded[k].items())[0][1] if len(loaded[k])>0 else None
                    cl2=str2Class(kk["____cls"]) if kk is not None and "____cls" in kk else cl
                    # print("bg")
                    # print(len(loaded[k].items()))
                    # print(cl2)
                    repo={k2:cl2.import__(cl2(),v2) for k2,v2 in loaded[k].items()}
                    # print("FINbg")

                # print(k)
                # print(repo)
                # setattr(ol,k,repo)
            else:
                if isinstance(get_origin(v)(),Base):
                    cl=get_origin(v)
                    kk=loaded[k]
                    cl2=str2Class(kk["____cls"]) if kk is not None and "____cls" in kk else cl
                    repo=cl2.import__(cl2(),loaded[k])
                else:
                    try:
                        repo=loaded[k]
                    except Exception as e:
                        print(loaded)
                        print(k)
                        print(v)
                        print(get_origin(v))
                        print(cls.__name__)
                        print(ol)
                        raise e
                # setattr(ol,k,repo)
            # print(cls.__name__)
            # if cls.__name__=="BaseSuperviseProject":
            #     print(k)
            #     print(repo)
            setattr(ol,k,repo)
        for k in annot2:
            # print("kk",k)
            repo=loaded[k]
            # if cls.__name__=="BaseSuperviseProject":
            #     print(k)
            #     print(repo)
            setattr(ol,k,repo)
        return ol

    @classmethod 
    def import__(cls,ol,loaded):
       return cls.import___(cls,ol,loaded)

    @classmethod 
    def _import(cls,loaded):
        return loaded

    @classmethod
    def Import(cls,ID,
             repertoire=None,
             ext=None,
             path=os.getcwd(),
             delim="/",
             loadArgs={},
            **xargs):
        loaded=cls.Load(ID,repertoire,
                                    ext,
                                    path,delim,suffix=cls.EXPORTABLE_SUFFIX,**loadArgs,**xargs)

        ol=cls()
        ol=cls.import__(ol,loaded)

        # if cls.__name__ == Base.__name__: 
        #     pass
        # else:
        #     # print(cls.__base__)
        #     try:
        #         papa = cls.__base__.Export(obj,save=False)
        #     except:
        #         papa = {}
        # # print(loaded)
        # ll=cls._import(loaded)
        # for k,v in ll.items():
        #     setattr(ol,k,v)
        return cls._import(ol)

    @classmethod
    def __export(cls,obj):
        # print(type(obj))
        # print(isinstance(type(obj),Base))
        if isinstance(obj,list) or isinstance(obj,tuple):
            return [ cls.__export(i) for i in obj ]
        elif isinstance(obj,dict):
            return {k:cls.__export(v) for k,v in obj.items()}
        if isinstance(obj,Base):
            rep=obj.export(save=False)
            rep["____cls"]=obj.__class__.__name__
            # rep.update({"____cls":obj.__class__.__name__})
            return rep
        return obj

    @classmethod
    def _export(cls,obj,*args,**xargs):
        try:
            rpi=cls.EXPORTABLE
            if "underscore" in cls.EXPORTABLE_ARGS:
                rpi=["_"+i for i in rpi]
            rep=takeInObjIfInArr(rpi,obj)
        except Exception as e:
            print(rpi)
            print(cls.__name__)
            print(obj)
            raise e
        rep={ k:cls.__export(v) 
                for k,v in rep.items() 
            }
        rep["____cls"]=cls.__name__
        return rep

    @staticmethod
    def Export__(cls,obj,save=True,saveArgs={}):
        kk=False
        # print(obj)
        if cls.__name__ == Base.__name__: 
            papa={}
        else:
            # print(cls.__base__)
            # try:
            papa={}
            # argus=get_default_args(cls.Export)
            for i in cls.__bases__:
                if hasattr(i,"Export"):
                    # if "me" in argus:
                    #     print(cls.__name__)
                    #     print(argus)
                    #     print(i.__name__)
                    #     if argus["me"] == i.__name__ and False:
                    #         continue
                    op=i.Export(obj,save=False)
                    papa=merge_two_dicts(papa,op)
            # except Exception as e:
            #     papa = {}
            #     # if cls.__name__  == "BaseSupervise":
            #     print(cls.__name__)
            #     print(obj)
            #     raise e

        # print(cls.__name__)
        rep  = cls._export(obj)
        rep  = merge_two_dicts(papa,rep)
        # print(rep.keys())

        if save:
            # print(rep)
            opts=merge_dicts(dict(ID=obj.ID,suffix=cls.EXPORTABLE_SUFFIX),saveArgs)
            return cls.Save(rep,**opts)
        return rep
    @classmethod
    def Export(cls,obj,save=True,saveArgs={}):
        return cls.Export__(cls,obj,save=save,saveArgs=saveArgs)

    def export(self,save=True,*args,**xargs):
        # print(self.__class__.__name__)
        # print(self)
        return self.__class__.Export(self,save,*args,**xargs)

    def __repr__(self,ind=1):
        nt="\n"+"\t"*ind
        stri="[[{}]"+nt+"ID : {}]"
        return stri.format(getClassName(self),self.ID)

    def __getattr__(self,a):
        if has_method(self,"_"+a): return getattr(self,"_"+a,None)
        else: raise AttributeError(a)

class Datas(Base):
    EXPORTABLE=["X","y","namesY"]
    def __init__(self,X=None,y=None,namesY=None,ID=None):
        super().__init__(ID)
        self.X=X
        self.y=y
        self.namesY=namesY
    
    @property
    def isMultiLabel(self):
        return np.ndim(self.y) > 1

    def get(self,withNamesY=True):
        return [self.X,self.y,self.namesY] if withNamesY else [self.X,self.y]

    def __repr__(self,ind=1):
        txt=super().__repr__(ind)
        t="\t"*ind
        nt="\n"+t
        stri=txt[:-1]+nt+"X : {},"+nt+"y : {},"+nt+"namesY : {}]"
        return stri.format(np.shape(self.X) if self.X is not None else None,np.shape(self.y) if self.y is not None else None,self.namesY)



class DatasSupervise(Base):
    EXPORTABLE=["dataTrain","dataTest"]
    def __init__(self,dataTrain:Datas=None,dataTest:Datas=None,ID=None):
        super().__init__(ID)
        self.dataTrain=dataTrain
        self.dataTest=dataTest

    @staticmethod
    def from_XY_Train_Test(X_train,y_train,X_test,y_test,namesY,ID=None):
        return DatasSupervise(Datas(X_train,y_train,namesY),
                              Datas(X_test,y_test,namesY),ID)
    def get(self,deep=False,optsTrain={},optsTest={}):
        if deep:
            return [*self.dataTrain.get(**optsTrain),*self.dataTest.get(**optsTest)]
        return [self.dataTrain,self.dataTest]

    def __repr__(self,ind=2):
        txt=super().__repr__(ind=ind)
        nt="\n"+"\t"*ind
        stri=txt[:-1]+nt+"dataTrain : {},"+nt+"dataTest : {}]"
        return stri.format(securerRepr(self.dataTrain,ind+2),
                            securerRepr(self.dataTest,ind+2))
class Models(Base):
    EXPORTABLE=["models","namesModels","mappingNamesModelsInd","indNamesModels"]
    def __init__(self,models=None,ID=None):
        super().__init__(ID)
        self.models=models
        self.init()
        self.initModelsAndNames()
        
    def init(self):
        self.namesModels=None
        self.indNamesModels=None
        self.mappingNamesModelsInd=None

    def initModelsAndNames(self):
        if self.models is not None:
            self.namesModels=np.array(uniquify([getClassName(i) for i in self.models]))
            self.indNamesModels=np.array(rangel(len(self.models)))
            self.mappingNamesModelsInd=dict(zip(self.namesModels,self.indNamesModels))
            
    def getIndexFromNames(self,arr,returnOK=True):
        arr=flatArray(arr)
        return [self.mappingNamesModelsInd[i] if isStr(i) else i for i in arr] if returnOK else None
    
    def __repr__(self,ind=1):
        txt=super().__repr__(ind)
        nt="\n"+"\t"*ind
        stri=txt[:-1]+nt+"models : {},"+nt+"namesModels : {},"+nt+"indNamesModels : {},"+nt+"mappingNamesModelsInd : {}]"
        return stri.format(securerRepr(BeautifulList(self.models),ind+1),
                        self.namesModels,
                        securerRepr(BeautifulList(self.indNamesModels) if self.indNamesModels is not None else self.indNamesModels,ind+1),
                        securerRepr(BeautifulDico(self.mappingNamesModelsInd),ind+1))


class Metric(Base):
    EXPORTABLE=["metric","scorer","metricName"]
    def __init__(self,metric=None,scorer=None,scorerToo=True,greaterIsBetter=True,ID=None,**xargs):
        super().__init__(ID)
        if metric is not None:
            self.metric=metric
            if isStr(metric):
                self.metric=get_metric(metric)
            else:
                self.metric=metric
            self.metricName=self.metric.__repr__()
            self.scorer=scorer
            if scorerToo and scorer is None:
                if isStr(metric):
                    self.scorer = get_scorer(metric)
                else:
                    self.scorer = make_scorer(self.metric,greaterIsBetter=greaterIsBetter,**xargs)

    def __repr__(self,ind=1):
        txt=super().__repr__(ind)
        nt="\n"+"\t"*ind
        if isStr(self.metric):
           return txt[:-1]+nt+"metric : {}]".format(self.metric) 
        return txt

            

class CvOrigSorted(Base):
    EXPORTABLE=["original","sorted"]
    """docstring for CvOrigSorted"""
    def __init__(self, original=None, sorted_=None, ID=None):
        super().__init__(ID)
        self.original = original
        self.sorted = sorted_
        

class CvResultatsTrValOrigSorted(Base):
    """docstring for CvResultatsPreds"""
    EXPORTABLE=["Tr","Val"]
    def __init__(self,tr:CvOrigSorted=None,val:CvOrigSorted=None, ID=None):
        super().__init__(ID)
        self.Tr=tr
        self.Val=val

class CvResultatsTrVal(Base):
    """docstring for CvResultatsPreds"""
    EXPORTABLE=["Tr","Val"]
    def __init__(self,tr=None,val=None, ID=None):
        super().__init__(ID)
        self.Tr=tr
        self.Val=val

class CvSplit(Base):
    """docstring for CvSplit"""
    EXPORTABLE=["train","validation","all_"]
    def __init__(self, train=None, validation=None, all_=None, ID=None):
        super().__init__(ID)
        self.train = train
        self.validation = validation
        self.all_=all_

    @staticmethod
    def fromCvSplitted(cv):
        return CvSplit([i[0] for i in cv],[i[1] for i in cv],cv)

class CvResultatsCvValidate(Obj): pass
class CvResultatsScores(CvResultatsTrVal): pass

class CvResultatsPreds(CvResultatsTrValOrigSorted): pass

class CvResultatsDecFn(CvResultatsTrValOrigSorted): pass
class CvResultats(Base):
    """docstring for CvResultats"""
    EXPORTABLE=["preds","scores","cv_validate","decFn"]
    def __init__(self, preds:CvResultatsPreds=None, scores=None, cv_validate=None, decFn=None, ID=None):
        super().__init__(ID)
        self.preds=preds
        self.scores=scores
        self.cv_validate=cv_validate
        self.decFn=decFn
        
from typing import Iterable
from typing import Dict, Tuple, Sequence, List

class CrossValidItem(CvResultatsTrValOrigSorted):
    EXPORTABLE=["cv","resultats","splitted","args"]
    def __init__(self,ID:str=None,cv:CvSplit=None,resultats:Dict[str,CvResultats]={},args:Dict=None):
        if cv is None:
             super().__init__()
        else:
            cvvTr=cv.train
            cvvVal=cv.validation
                               
            cvvTrCon=np.argsort(np.concatenate(cvvTr))
            cvvValCon=np.argsort(np.concatenate(cvvVal))

            super().__init__(CvOrigSorted(cvvTr,cvvTrCon),
                             CvOrigSorted(cvvVal,cvvValCon),ID)


        # self.name=name
        self.cv=cv
        self.splitted=cv
        self.resultats=resultats
        self.args=args

    def __repr__(self,ind=1):
        # txt="\n"
        txt=super().__repr__(ind=ind)
        nt="\n"+"\t"*ind
        stri=txt[:-1]+nt+"cv_ : {}"+nt+"resultats : {}"+nt+"args : [...]"
        #securerRepr(BeautifulDico(self.args),ind)
        return stri.format(securerRepr(self.cv,ind=ind+1),
            securerRepr(BeautifulDico(self.resultats),ind=ind+1))

class CrossValid(Base):
    EXPORTABLE=["cv","parallel","random_state","shuffle","classifier","recreate","metric","models","nameCV","argu"]
    def __init__(self,cv=None,classifier=None,metric:Metric=None,nameCV=None,parallel=True,random_state=42,
                 shuffle=True,recreate=False,models=None,namesMod=None):
        super().__init__(nameCV)
        self.cv=cv
        self.parallel=parallel
        self.random_state=random_state
        self.shuffle=shuffle
        self.classifier=classifier
        self.recreate=recreate
        self.metric=metric
        self.models=models
        self.nameCV = self.ID
        self.namesMod=namesMod
        self.argu=dict(cv=cv,random_state=random_state,shuffle=shuffle,classifier=classifier,
            nameCV=self.nameCV,namesMod=namesMod,recreate=recreate,parallel=parallel,metric=metric,models=models)
        self.cv=CrossValid.getCV(self)
        
    @staticmethod
    def getCV(self):
        if self.cv is None: return None
        if isinstance(self.cv,check_cv2): cv=cv
        elif isInt(self.cv): cv = check_cv2(self.cv,classifier=self.classifier,random_state=self.random_state,shuffle=self.shuffle)
        else: cv=self.cv
        return cv

    def checkOK(self,names):
        if (not self.recreate) and (self.nameCV in names):
            a=input("[computeCV] name '{}' is already take, recreate ? (y/N)".format(self.nameCV))
            if a=="y":
                return True
            else:
                nn=newStringUniqueInDico(name,dict(zip(names,[None]*len(names))))
                with ShowWarningsTmp():
                    warnings.warn("[CrossValid checkOK] name '{}' is already take, mtn c'est '{}'".format(self.nameCV,nn))
                self.nameCV=nn
        return True
            
    def computeCV(self,X,y,**xargs):
        self.cv = self.cv.split(X,y)
        self.cv=CvSplit.fromCvSplitted(self.cv)
        cv=self.cv
        cvo=self.crossV(X,y,**xargs)
        return (self.nameCV,CrossValidItem(self.nameCV,cv,cvo,self.argu))

    def crossV(self,X,y,verbose=0,n_jobs=-1,**xargs):
        cv=self.cv
        cvv=cv
        metric=self.metric.scorer
        parallel=self.parallel
        models=self.models
        namesMod=[i.__class__.__name__ for i in models] if self.namesMod is None else self.namesMod
        #models=ifelse(models,models,self.models)
                           
        cvvTr=cvv.train
        cvvVal=cvv.validation
                           
        cvvTrCon=np.argsort(np.concatenate(cvvTr))
        cvvValCon=np.argsort(np.concatenate(cvvVal))
        
        resu2=[cross_validate(mod ,X,y,return_train_score=True,
                            return_estimator=True,cv=cvv.all_,n_jobs=ifelse(parallel,n_jobs),verbose=verbose,scoring=metric) for mod in models]

        preduVal=[[i.predict(X[k]) for i,k in zipl(resuI["estimator"],cvvVal) ] for resuI in resu2]
                           
        preduuVal=[np.concatenate(preduI)[cvvValCon] for preduI in preduVal]
        
        scoreVal = [resuI["test_score"] for resuI in resu2]
        
        preduTr=[[i.predict(X[k]) for i,k in zipl(resuI["estimator"],cvvTr) ] for resuI in resu2]
                           
        preduuTr=[np.concatenate(preduI)[cvvTrCon] for preduI in preduTr]
        
        scoreTr = [resuI["train_score"] for resuI in resu2]
        
        decVal=[[getDecFn(i,X[k]) for i,k in zipl(resuI["estimator"],cvvVal) ] for resuI in resu2]
        decVal2=[concatenateDecFn(preduI,cvvValCon) for preduI in decVal]

        resul={}
        for i,name in enumerate(namesMod):
            resul[name]=CvResultats(CvResultatsPreds(CvOrigSorted(preduTr[i],preduuTr[i]),
                                                      CvOrigSorted(preduVal[i],preduuVal[i])),
                                     CvResultatsScores(scoreTr[i],scoreVal[i]),
                                     CvResultatsCvValidate(value=resu2[i]),
                                     CvResultatsDecFn(None,
                                                     CvOrigSorted(decVal[i],decVal2[i])) )

        return resul
        
class BaseSupervise(Base):
    # @abstractmethod
    EXPORTABLE=["datas","models","metric","cv","nameCvCurr"]
    EXPORTABLE_ARGS=dict(underscore=True)
    def __init__(self,ID=None,datas:DatasSupervise=None,
                    models:Models=None,metric:Metric=None,
                    cv:Dict[str,CrossValidItem]={},nameCvCurr=None,
                    *args,**xargs):
        super().__init__(ID)
        self._datas=datas
        self._models=models
        self._metric=metric
        self._cv=cv
        self._nameCvCurr=nameCvCurr
        self.init()
    
    def init(self):
        # self._cv={}
        pass
        # self._nameCvCurr=None
        

    def setDataTrainTest(self,X_train=None,y_train=None,X_test=None,y_test=None,namesY=None):
        #self.setDataX(X_train,X_test)
        self._datas=DatasSupervise.from_XY_Train_Test(X_train,y_train,
                                          X_test,y_test,
                                          namesY)

    def setModels(self,models):
        self._models=Models(models)

    def setMetric(self,metric):
        self._metric=Metric(metric)

    def computeCV(self,cv=5,random_state=42,shuffle=True,classifier=True,
                 nameCV=None,recreate=False,parallel=True,metric=None,
                 models=None):
        if models is not None:
            resu=self.getIndexFromNames(models)
            models=np.array(self.models)[resu]
            namesMod=np.array(self._models.namesModels)[resu]
        else:
            models=self.models
            namesMod=self._models.namesModels
        if metric is None:
            metric=self.scorer
        cv=CrossValid(cv=cv,random_state=random_state,shuffle=shuffle,classifier=classifier,
                 nameCV=nameCV,recreate=recreate,parallel=parallel,metric=metric,
                 models=models,namesMod=namesMod)
        cv.checkOK(list(self.cv.keys()))
        resu=cv.computeCV(self.X_train,self.y_train)
        self._cv[resu[0]]=resu[1]
        self._nameCvCurr=resu[0]
        
    
    @staticmethod
    def plan():
        print("PLAN:")
        print("\tCREATE_OR_GET WITH PROJECT, OTHERWISE")
        print("\t\tCLASS_STUDY.addOrGet(ID_STUDY)")
        print("\tSET DATA:")
        print("\t\tWith X,y (train and test):")
        print("\t\t\t"+"[Name_Variable_Study].setDataTrainTest(X_train, y_train, X_test, y_test, namesY)")
        print("\t\tWith ProjectData ID:")
        print("\t\t\t"+"[Name_Variable_Study].setDataTrainTest(id_=\"DATA_ID\")")
        print("\tPREPROCESS ProjectData (If Data from Project, you have maybe to preprocess this data for the study (not preprocess for the ML) :")
        print("\t\t[Name_Variable_Study].proprocessDataFromProject(fn=FUNCTION_TO_CALL)")
        print("\t\tthe signature of the FUNCTION_TO_CALL is  FUNCTION_TO_CALL(X_train,y_train, X_test, y_test, namesY) and return a list or a tuple with [X_train,y_train, X_test, y_test, namesY]")
        print("\tSET MODELS:")
        print("\t\t"+"[Name_Variable_Study].setModels(models=MODELS)")
        print("\t\t"+"MODELS MUST RESPECT THE INTERFACE OF SKLEARN (fit, transform, predict, (predict_proba or decision_function))")
        print("\tCOMPUTE CROSS-VALIDATION:")
        print("\t\t[Name_Variable_Study].computeCV()")
        print("\tSAVE the study (if in studyProject preferes save with the project not the study):")
        print("\t\t[Name_Variable_Study].save()")

    def help(self):
        getStaticMethodFromObj(self,"plan")()

    def getIndexFromNames(self,m):
        return self.models.getIndexFromNames(m)
        
    @property
    def isMultiLabel(self):
        return self.train_datas.isMultiLabel
    
    @property
    def modelsObj(self):
        return self._models
    
    @property
    def metric(self):
        return self._metric.metric
    
    @property
    def datas(self):
        return self._datas
    
    @property
    def train_datas(self):
        return self.datas.dataTrain
    
    @property
    def test_datas(self):
        return self.datas.dataTest
    
    @property
    def X_train(self):
        return self.train_datas.X
    
    @property
    def y_train(self):
        return self.train_datas.y
    
    @property
    def X_test(self):
        return self.test_datas.X
    
    @property
    def y_test(self):
        return self.test_datas.y
    
    @property
    def models(self):
        return self._models.models
    
    @property
    def namesModels(self):
        return self._models.namesModels
    
    @property
    def namesModels(self):
        return self._models.indNamesModels
    
    @property
    def namesModels(self):
        return self._models.mappingNamesModelsInd
    
    @property
    def namesY(self):
        return self.train_datas.namesY

    @classmethod
    def addOrGet(cls,id_,recreate=False,clone=False,deep=True,*args,**xargs):
        # cls=self.__class__
        def clonee(rrt):
            return getStaticMethodFromObj(rrt,"clone")(rrt,deep=deep)
        def recreatee():
            rrt=cls(ID=id_)
            if clone :
                rrt=clonee(rrt)
            # res =self.add(rrt)
            return rrt
        def cloneStudy(res):
            ru=res
            ru=clonee(ru)
            # self._studies[id_]=ru
            return ru
        if recreate:
            res=recreatee()
        else:
            res=cls.Load(id_,*args,**xargs)
            if resu is None:
                res=recreatee()
            else:
                if clone:
                    res=cloneStudy(res)
        return res

    def __repr__(self,ind=1,orig=False):
        if orig:
            return object.__repr__(self)
        txt=super().__repr__(ind=ind)
        nt="\n"+"\t"*ind
        stri=txt[:-1]+nt+"datas : {},"+nt+"models : {},"+nt+"metric : {},"+nt+"cv : {},"+nt+"nameCvCurr : {}]"
        # print(stri.format(self._datas,self._models,self._metric,self._cv,self._nameCvCurr))
        return stri.format(securerRepr(self._datas,ind=ind+2),
            securerRepr(self._models,ind+2),
            securerRepr(self._metric,ind+2),
            securerRepr(BeautifulDico(self._cv),ind+1,ademas=2),self._nameCvCurr)

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
    
def concatenateDecFn(ll,k):
    #print(np.ndim(ll))
    #print(np.ndim(ll[0]))
    if np.ndim(ll[0])==3:
        dfr=np.concatenate([i for i in ll],axis=1)
        dfr=[i[k] for i  in dfr]
        return dfr
    return np.concatenate(ll)[k]    
 