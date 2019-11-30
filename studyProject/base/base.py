from ..utils import studyDico, isStr, get_metric, ifelse, randomString, uniquify, \
                    getClassName, rangel, SaveLoad, merge_two_dicts, \
                    newStringUniqueInDico, check_cv2, Obj, mapl, ifNotNone, has_method ,\
                    isInt, zipl, BeautifulDico,BeautifulList, getStaticMethodFromObj,\
                    takeInObjIfInArr, convertCamelToSnake, getAnnotationInit, securerRepr, merge_dicts,\
                    namesEscape,listl, T, F, StudyClass, isPossible, get_default_args
from sklearn.metrics import make_scorer, get_scorer
from sklearn.model_selection import cross_validate
import numpy as np
import copy
import os
from tqdm import tqdm
import shutil 
import warnings
import pandas as pd
from interface import implements, Interface
from abc import ABCMeta, abstractmethod, ABC
from ..version import __version__
from studyPipe import *

from plotly_study.subplots import make_subplots
from ..viz.viz import vizHelper
import inspect
from typing import Dict
from ..utils import isinstanceBase, isinstance, make_tarfile, read_tarfile, StudyList,studyList, TMP_DIR, showWarningsTmp
from ..utils.speedMLNewMethods import create_speedML
# from ..viz import StudyViz_Datas
# from typing import get_origin
# class ImportExportLoadSaveClone(Interface):
try:
    from typing import TypeVar, _GenericAlias
except:
    from typing import GenericMeta as _GenericAlias
    class TypeVar:
        def __init__(self,*args,**xargs):pass
import sys

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
def get_args_typing(l):
    try:
        rep=l.__args__
    except:
        rep=l
    if rep is None:
        rep=[TypeVar("blabla"),TypeVar("blabla")]
    return rep
class Base(object):
    DEFAULT_PATH="__studyFiles"
    DEFAULT_PATH_FTS="__toSave"
    DEFAULT_REP="study_"
    DEFAULT_EXT=".study_"
    DEFAULT_DIRS="dirs.txt"
    EXPORTABLE=["ID"]
    EXPORTABLE_SUFFIX="EXP"
    EXPORTABLE_ARGS={}


    def __getstate__(self): return self.__dict__.copy()
    def __setstate__(self, d): self.__dict__.update(d)
    # def __getstate__(self):
    #     try:
    #         state = super().__getstate__()
    #     except AttributeError:
    #         state = self.__dict__.copy()

    #     if type(self).__module__.startswith('sklearn.'):
    #         return dict(state.items(), _sklearn_version=__version__)
    #     else:
    #         return state

    # def __setstate__(self, state):
    #     if type(self).__module__.startswith('sklearn.'):
    #         pickle_version = state.pop("_sklearn_version", "pre-0.18")
    #         if pickle_version != __version__:
    #             warnings.warn(
    #                 "Trying to unpickle estimator {0} from version {1} when "
    #                 "using version {2}. This might lead to breaking code or "
    #                 "invalid results. Use at your own risk.".format(
    #                     self.__class__.__name__, pickle_version, __version__),
    #                 UserWarning)
    #     try:
    #         super().__setstate__(state)
    #     except AttributeError:
    #         self.__dict__.update(state)
    # viz=


    def init(self,*args,**xargs):
        pass
    #include all function even if in dad
    @staticmethod
    def __viz_viz__(self,selfo):
        # print(self.__name__)
        try:
            n=self.__name__
            # print("Study_"+n+"_Viz")
            # print("Study_"+n+"_Viz")
            rep=str2Class("Study_"+n+"_Viz")(selfo)
            # print("ok")
        except:
            rep=None
            # n2=self
            if hasattr(self,"__bases__"):
                # print(self.__bases__)
                for i in self.__bases__[::-1]:
                    rep=Base.__viz_viz__(i,selfo)
                    if rep is not None:
                        break
            elif hasattr(self,"__base__"):
                # print(self.__base__)
                rep= Base.__viz_viz__(self.__base__,selfo)
        # print(rep)
        return rep

    @property
    def viz(self):
        # print(self.__class__.__name__)
        return Base.__viz_viz__(self.__class__,self)


    # @property
    # def viz(self):
    #     try:
    #         # print("Study_"+self.__class__.__name__+"_Viz")
    #         rep=str2Class("Study_"+self.__class__.__name__+"_Viz")(self)
    #     except:
    #         rep=None
    #         # # # # 
    #         # # # # # # #####  
    #         # # # # # # #####   
    #         # # # #     # # #  
    #         if hasattr(self.__class__,"__bases__"):
    #             for i in self.__class__.__bases__:
    #                 # print("ici")
    #                 # print(i)
    #                 g=super(i,self)
    #                 # print(g)
    #                 # print(dir(g))
    #                 # print("viz" in dir(g) or "viz" in dir(i))
    #                 rep=getattr(g,"viz") if "viz" in dir(g) or "viz" in dir(i) else None
    #                 # rep=str2Class("Study_"+i.__name__+"_Viz")(self)
    #                 if rep is not None:
    #                     break
    #         elif hasattr(self.__class__,"__base__"):
    #             # print("ici2")
    #             # print(self.__class__)
    #             # print("Study_"+self.__class__.__base__.__name__+"_Viz")
    #             rep=str2Class("Study_"+self.__class__.__base__.__name__+"_Viz")(self)
    #     return rep
    

    @classmethod
    def get_repertoire(cls,repertoire,suffix=None,chut=True):
        suffix=cls.EXPORTABLE_SUFFIX if suffix is None else suffix
        if repertoire is None:
            if cls.__name__ != Base.__name__ and cls.DEFAULT_REP!= Base.DEFAULT_REP:
                repertoire=cls.DEFAULT_REP+"_"+suffix
            else:
                repertoire=cls.DEFAULT_REP+convertCamelToSnake(cls.__name__)+"_"+suffix
                if not chut:
                    warnings.warn("\n[Base save] repertoire est non spécifié, repertoire = {} ".format(repertoire))
        return repertoire

    @classmethod
    def get_ext(cls,ext,suffix=None,chut=True):
        suffix=cls.EXPORTABLE_SUFFIX if suffix is None else suffix
        if ext is None:
            if cls.__name__ != Base.__name__ and cls.DEFAULT_REP!= Base.DEFAULT_REP:
                ext =cls.DEFAULT_EXT+"."+suffix
            else:
                ext=cls.DEFAULT_EXT+convertCamelToSnake(cls.__name__)+"."+suffix
                if not chut:
                    warnings.warn("\n[Base save] ext est non spécifié, ext = {} ".format(ext))
        return ext

    def __init__(self,ID:str=None):
        self.ID=ifelse(ID is None,lambda:randomString(),lambda:ID)() 
        object.__setattr__(self, "papa", None)
        object.__setattr__(self, "attr", None)
        #self.papa=self

    def clone(self,ID=None,newIDS=False,deep=True,*args,**xargs):
        return self.__class__.Clone(self,ID,newIDS=newIDS,deep=deep,*args,**xargs)
        
    @staticmethod
    def Clone(self,ID=None,deep=True,newIDS=False,normalNEW=True,preservePAPA=False):
        ID = ifelse(ID is not None,ID,self.ID)
        #print( name)
        if deep:
            exported=self.export(save=False)
            if normalNEW:
                l=self.__class__.import__(self.__class__(),exported,newIDS=newIDS)
            else:
                l=self.__class__.import__(self.__class__(),exported,newIDS=newIDS,normalNEW=False)

            l.ID=ID
            # l=self.__class__(ID)
            # l.__dict__ = merge_two_dicts(l.__dict__,self.__dict__.copy())
            # #print(l.__dict__)
            # l.ID=ID
            #print(l.__dict__)
            if preservePAPA and hasattr(self,"papa") and self.papa is not None:
                l.papa=self.papa
            return l
        else:
            me=copy.deepcopy(self)
            me.ID=ID
            if preservePAPA and hasattr(self,"papa") and self.papa is not None:
                me.papa=self.papa
            return me

    @classmethod
    def build_repertoire(cls,repertoire,path=os.getcwd(),returnOK=True,dp=None,delim="/",
        fn=lambda repo:os.makedirs(repo),returnFn=False):
        dp=cls.DEFAULT_PATH if dp is None else dp 
        repo=delim.join([i for i in [path,dp,repertoire] if i != ""])
        repu=[]
        if not os.path.exists(repo):
            if returnOK:
                repu=repu+[fn(repo)]
        return [repo]+repu

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
                        fn=lambda repo:os.makedirs(repo),recreate=False,chut=True,
                        returnFn=False):
        if fn is None:
            fn=lambda repo:os.makedirs(repo)
        repos=cls.build_repertoire(repertoire,path=path,dp=dp,delim=delim,fn=fn,returnFn=returnFn)
        if isinstance(repos,list):
            repo=repos[0]
        return (repos,
                     cls.build_ext(repo,ext,ID,delim,recreate,chut))
    
    @staticmethod
    def __listFilesAndTar(dirFiles,sv,patho=None):
        ooo=[sv]
        # print(dirFiles)
        # print(sv)
        if dirFiles is not None:
            ooo=[sv]+list(dirFiles)
            tt="\n".join(dirFiles)
            io=TMP_DIR()
            iof=io.get()
            # print(iof)
            # print(iof+"/"+Base.DEFAULT_DIRS)
            with open(iof+"/"+Base.DEFAULT_DIRS,"w") as f:
                f.write(tt)
            u=iof+"/"+Base.DEFAULT_DIRS
            ooo=[u]+ooo
        patho=Base.DEFAULT_PATH+"/"+Base.DEFAULT_PATH_FTS+"/"+randomString() if patho is None else patho
        # print(patho)
        # print(ooo)
        pathoTar=make_tarfile(patho,ooo,ext="bz2")
        if dirFiles is not None:
            io.delete()
        return pathoTar

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
             addExtension=True,
             fnBuildRepExt=None,
             dirAdded=[],
             **xargs):
        ID=self.ID if ID is None else ID
        if noDefaults:
            repertoire=""
            ext=""


        repertoire,ext=cls.get_rep_ext(repertoire,ext,chut=chut)

        repo,(ok,filo)=cls.build_rep_ext(repertoire,ext,ID,dp=cls.DEFAULT_PATH if not noDefaults else "",
                                            chut=chut,recreate=recreate,fn=fnBuildRepExt)
        # print(ok)
        if ok:
            dirTmp=None
            if len(dirAdded) >0:
                dirTmp=dirAdded
            filos=SaveLoad.save(self,filo,chut=chut,addExtension=addExtension,fake=True,**xargs)
            # print(filos)
            filo=filos+".partial"
            patho=SaveLoad.save(self,filo,chut=chut,addExtension=False,**xargs)
            # print(patho)
            if patho is None:
                raise Exception("ERROR patho")
            try:
                Base.__listFilesAndTar(dirTmp,patho,filos)
            finally:
                os.remove(patho)
    
    def save(self,
             repertoire=None,
             ext=None,
             ID=None,
             path=os.getcwd(),
             delim="/",
             recreate=False,
             addExtension=True,
             dirAdded=[],
             **xargs):
        self.__class__.Save(self,ID,repertoire,ext,path,delim,recreate,dirAdded=dirAdded,**xargs)
    
    @classmethod
    def Load(cls,ID,
             repertoire=None,
             ext=None,
             path=os.getcwd(),
             delim="/",
             suffix="",
             chut=True,
             noDefaults=False,
             addExtension=True,
            **xargs):
        # print("coucou")
        if noDefaults:
            repertoire=""
        if repertoire is None:
            if cls.__name__ != Base.__name__ and cls.DEFAULT_REP!= Base.DEFAULT_REP:
                repertoire =cls.DEFAULT_REP+"_"+suffix
                if suffix=="":
                    repertoire=cls.DEFAULT_REP
            else:
                repertoire=cls.DEFAULT_REP+convertCamelToSnake(cls.__name__)+"_"+suffix
                if not chut:
                    warnings.warn("\n[Base load] repertoire est non spécifié, repertoire = {} ".format(repertoire)) 
                if suffix=="":
                    repertoire=cls.DEFAULT_REP+convertCamelToSnake(cls.__name__)
        if noDefaults:
            ext=""
        if ext is None:
            if cls.__name__ != Base.__name__ and cls.DEFAULT_REP!= Base.DEFAULT_REP:
                ext =cls.DEFAULT_EXT+"."+suffix
                if suffix=="":
                    ext=cls.DEFAULT_EXT
            else:
                ext=cls.DEFAULT_EXT+convertCamelToSnake(cls.__name__)+"."+suffix
                if not chut:
                    warnings.warn("\n[Base load] ext est non spécifié, ext = {} ".format(ext))
                if suffix=="":
                    ext=cls.DEFAULT_EXT+convertCamelToSnake(cls.__name__)
        dp=cls.DEFAULT_PATH if not noDefaults else "" 
        repo=delim.join([i for i in [path,dp,repertoire] if i != ""])
        # print(ID)
        # print(ext)
        filo=delim.join([i for i in [repo,ID+ext] if i != ""]) #repo+delim+ID+ext
        # print(filo)
        filo=SaveLoad.load(filo,addExtension=addExtension,chut=chut,fake=True,**xargs)
        # print(filo,os.path.isfile(filo))
        if not os.path.isfile(filo):
            if not chut:
                warnings.warn("\n[Base load] {} n'exite pas ".format(filo))
            return None
        try:
            # print(filo)
            # yyy=SaveLoad.load(filo,addExtension=addExtension,chut=chut,fake=True,**xargs)
            yyy=filo
            res=read_tarfile(yyy,ext="bz2")
            gg4={}
            # print(res)
            # print(os.path.isfile(res+"/dirs.txt"))
            if os.path.isfile(res+"/dirs.txt"):
                with open(res+"/dirs.txt","r") as f:
                    gg=f.readlines()
                gg2=set([i.rstrip("\n\r") for i in gg])
                # print(gg2)
                hu=[]
                for j in gg2:
                    u=TMP_DIR()
                    uu=u.get()
                    hu.append(uu)
                    # print(res+"/"+j)
                    j2=os.path.basename(j)
                    shutil.move(res+"/"+j2+"/*", uu) 
                # print(gg2)
                gg4=dict(zip(gg2,hu))
                # print(gg4)
            filo=res+"/"+os.path.basename(yyy)+".partial"
            # print(filo)
            resu=SaveLoad.load(filo,addExtension=False,chut=chut,**xargs)
            return (resu,gg4)
        except Exception as e:
            # raise e
            resu=None
        return resu

 
    def addFileToSave(self):
        return []

    def addDirToSave(self):
        return []

    def restoreDir(self,logdir):
        return {}

    @staticmethod
    def import___(cls,ol,loaded,newIDS=False,
                    papaExport=[],forceInfer=False,dirs={},*args,**xargs):
        if loaded is None:
            return None
        if isinstance(loaded,dict) and len(loaded)==0:
            return {}
        if isinstance(loaded,list) and len(loaded)==0:
            return []
        if isinstance(loaded,tuple) and len(loaded)==0:
            return tuple()
        # argus=get_default_args(cls.import__)
        sameExport=False
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
                jj=i.EXPORTABLE==cls.EXPORTABLE
                sameExport|=jj
                ol=i.import__(ol,loaded,newIDS=newIDS,
                    papaExport=cls.EXPORTABLE if not jj else [],dirs=dirs,
                    *args,**xargs)
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
        if sameExport:
            return ol
        f=cls.EXPORTABLE
        ff=cls.EXPORTABLE_ARGS
        annot=getAnnotationInit(cls())
        if "underscore" in ff and ff["underscore"]:
            annot={("_"+k):v for k,v in annot.items() if k in f}
        else:
            annot={k:v for k,v in annot.items() if k in f}
        if "underscore" in ff and ff["underscore"]:
            annot2=["_"+i for i in f if "_"+i not in annot]
        else:
            annot2=[i for i in f if i not in annot]
        # if cls.__name__ in ["BaseSuperviseClassifProject",
        #                     "StudyClassifProject"]:
        #     print(cls)
        #     print(annot)
        #     print(annot2)
        #     print(papaExport)
        for k,v in annot.items():
            # print(k)
            if k in papaExport:
                # print(papaExport)
                continue    
            # if (k == "cv" or k == "_cv") and cls.__name__ in ["BaseSuperviseClassifProject",
            #                 "StudyClassifProject"]:
            #     print("ici2")
            #     print(k)
            #     print(v)
            #     print(cls.__name__)
            #     print(get_origin(v))
            #     print(type(loaded[k]))
            #     print(loaded[k])
            #     print(loaded)
            # print(k)
            # print(get_origin(v) is dict and isinstance(loaded[k],dict))
            if k not in loaded:
                repo=getattr(ol,k)
                setattr(ol,k,repo)
            if get_origin(v) is list and isinstance(loaded[k],list)  :
                if k not in loaded:
                    repo=getattr(ol,k)
                else:
                    cl=get_args_typing(v)[0] if isinstance(v,_GenericAlias) else v
                    if isinstance(cl,TypeVar) or not isinstance(cl(),Base):
                        repo=loaded[k]
                    else:
                        kk=loaded[k][0] if len(loaded[k])>0 else None
                        cl2=str2Class(kk["____cls"]) if kk is not None and "____cls" in kk else cl
                        if isinstance(cl(),cl2) and not forceInfer:
                            cl2=cl
                        repo=[ cl2.import__(cl2(),i,newIDS=newIDS,dirs=dirs,*args,**xargs) for i in loaded[k] ]
                # setattr(ol,k,repo)
            elif get_origin(v) is dict and isinstance(loaded[k],dict):
                # if k == "cv" or k == "_cv":
                #     print("ici")
                #     print(k)
                #     print(v)
                #     print(cls.__name__)
                if k not in loaded:
                    repo=getattr(ol,k)
                else:
                    cl=get_args_typing(v)[1] if isinstance(v,_GenericAlias) else v
                    if isinstance(cl,TypeVar) or not isinstance(cl(),Base):
                        repo=loaded[k]
                    else:
                        kk=list(loaded[k].items())[0][1] if len(loaded[k])>0 else None
                        cl2=str2Class(kk["____cls"]) if kk is not None and "____cls" in kk else cl
                        # print("bg±")
                        # print(len(loaded[k].items()))
                        # print(cl2)
                        if isinstance(cl(),cl2) and not forceInfer:
                            cl2=cl
                        # print(cl2) 
                        repo=studyDico({k2:cl2.import__(cl2(),v2,newIDS=newIDS,dirs=dirs,*args,**xargs) for k2,v2 in loaded[k].items()})
                    # print("FINbg")

                # print(k)
                # print(repo)
                # setattr(ol,k,repo)
            else:
                if isinstance(get_origin(v)(),Base):
                    if k not in loaded:
                        repo=getattr(ol,k)
                    else:
                        cl=get_origin(v)
                        kk=loaded[k]
                        cl2=str2Class(kk["____cls"]) if kk is not None and "____cls" in kk else cl
                        if isinstance(cl(),cl2) and not forceInfer :
                            cl2=cl
                        repo=cl2.import__(cl2(),loaded[k],newIDS=newIDS,dirs=dirs,*args,**xargs)
                else:
                    try:
                        # prinst(k)
                        # print(newIDS)
                        if k != "ID" or not newIDS:
                            if k not in loaded:
                                repo=getattr(ol,k)
                            else:
                                repo=loaded[k]
                        else:
                            repo=getattr(ol,k)
                    except Exception as e:
                        # pass
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

            setattr(ol,k,repo)
        for k in annot2:
            if k in papaExport:
                continue 
            # print("kk",k)
            if k != "ID" or not newIDS:
                if k not in loaded:
                    repo=getattr(ol,k)
                else:
                    repo=loaded[k]
                setattr(ol,k,repo)
        return ol

    @classmethod 
    def import__(cls,ol,loaded,newIDS=False,normalNEW=True,forceInfer=False,dirs={},*args,**xargs):
        # if normalNEW:
        ol.____IMPORT____=True
        rep=cls.import___(cls,ol,loaded,newIDS=newIDS,forceInfer=forceInfer,dirs=dirs,*args,**xargs)
        # rep=cls.import___(cls,cls(normal=False),loaded,newIDS=newIDS,forceInfer=forceInfer,*args,**xargs)
        if rep is not None and isinstance(rep,Base):
            rep.restoreDir(dirs)
        if hasattr(rep,"____IMPORT____"):
            delattr(rep,"____IMPORT____")
        return cls._import(rep)

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
             forceInfer=False,
            **xargs):
        # print(loadArgs)
        # print(xargs)
        loaded,dirs=cls.Load(ID,repertoire,
                                    ext,
                                    path,delim,suffix=cls.EXPORTABLE_SUFFIX,**loadArgs,**xargs)
        # print("dirs",dirs)
        if "__version__" in loaded:
            if loaded["__version__"] != __version__:
                warnings.warn(
                    "Trying to unpickle {0} from version {1} when "
                    "using version {2}. This might lead to breaking code or "
                    "invalid results. Use at your own risk.".format(
                        cls.__name__, loaded["__version__"], __version__),
                    UserWarning)
        ol=cls()
        ol=cls.import__( ol, loaded, forceInfer=forceInfer,dirs=dirs )

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
        return ol

    @classmethod
    def __export(cls,obj,dirAdded=[]):
        # print(type(obj))
        # print(isinstance(type(obj),Base))
        # print("ici2",dirAdded,cls)
        if isinstance(obj,list) or isinstance(obj,tuple):
            return [ cls.__export(i,dirAdded=dirAdded) for i in obj ]
        elif isinstance(obj,dict):
            return {k:cls.__export(v,dirAdded=dirAdded) for k,v in obj.items()}
        if isinstance(obj,Base):
            # print("__export",type(dirAdded))
            # print(obj)
            dirAdded.extend( obj.addFileToSave() )
            dirAdded.extend(  obj.addDirToSave() )
            # print(dirAdded)
            # print(obj.__class__.__name__)
            # print("__export END")
            rep=obj.export(save=False,dirAdded=dirAdded)
            rep["____cls"]=obj.__class__.__name__
            # rep.update({"____cls":obj.__class__.__name__})
            return rep
        return obj

    @classmethod
    def _export(cls,obj,papaExport=[],dirAdded=[],*args,**xargs):
        try:
            rpi=cls.EXPORTABLE
            if "underscore" in cls.EXPORTABLE_ARGS and cls.EXPORTABLE_ARGS["underscore"]:
                rpi=["_"+i for i in rpi]
            rep=takeInObjIfInArr(rpi,obj)
        except Exception as e:
            print(rpi)
            print(cls.__name__)
            print(obj)
            raise e
        # print("ici",dirAdded,cls)
        rep={ k:cls.__export(v,dirAdded=dirAdded) 
                for k,v in rep.items() if k not in papaExport
            }
        rep["____cls"]=cls.__name__
        return rep

    def __dir__(self):
        rep=super().__dir__()
        if len(self.__class__.EXPORTABLE)>0 and "underscore" in self.__class__.EXPORTABLE_ARGS and self.__class__.EXPORTABLE_ARGS["underscore"]:
            rep=self.__class__.EXPORTABLE+rep
        return rep

    @staticmethod
    def Export___(cls,obj,save=True,version=None,papaExport=[],dirAdded=[]):
        kk=False
        # print(obj)
        if cls.__name__ == Base.__name__: 
            papa={}
        else:
            # print(cls.__base__)
            # try:
            papa={}
            sameExport=False
            # argus=get_default_args(cls.Export)
            for i in cls.__bases__:
                if hasattr(i,"Export"):
                    # if "me" in argus:
                    #     print(cls.__name__)
                    #     print(argus)
                    #     print(i.__name__)
                    #     if argus["me"] == i.__name__ and False:
                    #         continue
                    jj=i.EXPORTABLE==cls.EXPORTABLE
                    sameExport|=jj
                    # print("export",i)
                    op=i.Export(obj,save=False,papaExport=cls.EXPORTABLE if not jj else [],dirAdded=dirAdded)
                    papa=merge_two_dicts(papa,op)
            # except Exception as e:
            #     papa = {}
            #     # if cls.__name__  == "BaseSupervise":
            #     print(cls.__name__)
            #     print(obj)
            #     raise e

        # print(\cls/\__name__)
        # print(dirAdded,cls)
        rep  = cls._export(obj,papaExport=papaExport,dirAdded=dirAdded)
        rep  = merge_two_dicts(papa,rep)
        dirAdded.extend(  obj.addDirToSave() )
        dirAdded.extend(  obj.addFileToSave() )
        return rep

    @staticmethod
    def Export__(cls,obj,save=True,saveArgs={},version=None,papaExport=[],dirAdded=[]):
        rep=cls.Export___(cls,vizGet(obj),papaExport=papaExport,save=save,dirAdded=dirAdded)
        if version is not None:
            rep["__version__"]=version
        # print(rep.keys())
        # print(dirAdded)
        if save:

            dirAdded=set(dirAdded)

            opts=merge_dicts(dict(ID=obj.ID,suffix=cls.EXPORTABLE_SUFFIX),saveArgs)
            return cls.Save(rep,dirAdded=dirAdded,**opts)
        return rep
    @classmethod
    def Export(cls,obj,save=True,version=None,saveArgs={},papaExport=[],dirAdded=[]):
        dirAdded=studyList(dirAdded) if not isinstance(dirAdded,StudyList) else dirAdded
        return cls.Export__(cls,obj,save=save,version=version,saveArgs=saveArgs,papaExport=papaExport,dirAdded=dirAdded)

    def export(self,save=True,dirAdded=[],*args,**xargs):
        dirAdded=studyList(dirAdded) if not isinstance(dirAdded,StudyList) else dirAdded
        # print(self.__class__.__name__)
        # print(self)
        return self.__class__.Export(self,save,version=__version__,dirAdded=dirAdded,*args,**xargs)

    def __repr__(self,ind=1):
        nt="\n"+"\t"*ind
        stri="[[{}]"+nt+"ID : {}]"
        return stri.format(getClassName(self),self.ID)

    def __getattribute__(self,a):
        rep=super().__getattribute__(a)
        if a != "papa" and not isinstance(getattr(type(self), a, None), property):
            if isinstance(rep,Base):
                object.__setattr__(rep, "papa", self)
                # rep.papa=self
                object.__setattr__(rep, "attr", a)
                # rep.attr=a
        return rep

    def __setattr__(self,a,b):
        if a in ["papa","attr"] and hasattr(self,a) and getattr(self,a) is None:
            raise Exception("papa and attr not setting (private of Base)")
        return object.__setattr__(self, a, b)

    def __getattr__(self,a):
        key=a
        if key.startswith('__') and key.endswith('__'):
            return super().__getattr__(key)
        if has_method(self,"_"+a): return getattr(self,"_"+a,None)
        else: raise AttributeError(a)
# factoryCls.register_class(Base)

# class NamesY(Base):

#     def __init__(self,Ynames=None,ID=None):
#         super().__init__(ID)
#         self.namesY=Ynames
#         self.init()

#     def init(self):
#         if self.namesY is not None:
#             if isinstance(self.namesY,list):
#                 pass
#     def check(self,y):
#         if not isinstance(y,pd.Series):
#             y=pd.Series(y)
#         val=y.value_counts().index.tolist()
#         return val

    # def fromSeries(self,s,index=False):
    #     if not index:
    #         if isinstance(self.namesY,list):

    #         s.replace(self.namesY)
            # {False:"Pas5",True:"5"}
import pandas_profiling_study as pdp
class edaCls:
    SECTIONS=["overview","variables","correlations","missing","sample"]
    def __init__(self,edas):
        self._edas=edas
    
    @property
    def all(self):
        return self.get()

    @property
    def overview(self):
        return self.get(["overview"])

    @property
    def variables(self):
        return self.get(["variables"])
    
    @property
    def correlations(self):
        return self.get(["correlations"])

    @property
    def missing(self):
        return self.get(["missing"])
    
    @property
    def sample(self):
        return self.get(["sample"])
    
    def get(self,sections=["overview","variables","correlations","missing","sample"]):
        return self._edas.change_sections(sections)

    def __repr__(self):
        return "eda, attribute available : "+", ".join(self.SECTIONS)

class Datas(Base):
    EXPORTABLE=["X","y","_eda","_prep"]
    # y must be a series or a dataframe

    def __init__(self,X=None,y=None,_eda=None,_prep=None,ID=None):
        super().__init__(ID)
        self.X=X
        self.y=y
        self._eda=_eda
        self._prep=_prep
        # self.init()

    def initEda(self):
        # print("initEDA",hasattr(self,"____IMPORT____"),self._eda,self.X)
        if not hasattr(self,"____IMPORT____"):
            eda=self._eda
            if eda is None and self.X is not None:
                with showWarningsTmp:
                    warnings.warn("""
                        creating EDA ProfileReport... (think to export the study(Project) !!)""")
            self._eda=eda if eda is not None else (None if self.X is None else pdp.ProfileReport(self.get(initial=True),sections=["overview","variables","correlations","missing","sample"]))
            if isinstance(self._eda ,pdp.ProfileReport):
                self._eda=edaCls(self._eda)
    def initPrep(self):
        _prep=self._prep
        # self._prep=_prep if _prep is not None else (None if self.X is None else prep(self))

    def init2(self):
        self.initEda()
        self.initPrep()

    def get(self,prep=True,withNamesY=False,concat=True,initial=False):
        if initial:
            return [self.X,self.y] if not concat else pd.concat([self.X,self.y],axis=1)
        if self.papa._prep is not None or self._prep is not None:
            # return self.prep.data
            return self.prep.getData()
        hj=[self.X,self.y]
        return hj if not concat else pd.concat(hj,axis=1)

    def __repr__(self,ind=1):
        txt=super().__repr__(ind)
        t="\t"*ind
        nt="\n"+t
        stri=txt[:-1]+nt+"X : {},"+nt+"y : {}]"
        return stri.format(np.shape(self.X) if self.X is not None else None,np.shape(self.y) if self.y is not None else None)

    @property
    def prep(self):
        # print("io",self.papa)
        if self.papa is None:
            if self._prep is None:
                self._prep=Dora(self.get(initial=True),output=self.y.name)
        else:
            # print("io2",prepI(self))
            self._prep=prepI(self)

        # if self._prep is None:
        #     self.initPrep()
        # if self._prep is None:
        #     raise Exception("prep not set")
        return self._prep

    @property
    def eda(self):
        if self._eda is None:
            self.initEda()
        if self._eda is None:
            raise Exception("eda not set")
        return self._eda

    #TODO: plotly chart in pdp
    def getEDA(self,concat=True, sections=["overview","variables","correlations","missing","sample"]):
        # if self.eda is None:
        #     with showWarningsTmp:
        #         warnings.warn("""
        #             creating ProfileReport...""")
        #     self.eda=pdp.ProfileReport(self.get(concat=concat),sections=["overview","variables","correlations","missing","sample"])
        return self.eda.get()

    def getEDA_Clues(self):
        d=self.papa.getEDA()
        return d

    def __dir__(self):
        d1=super().__dir__()
        d=dir(self.get())
        return [i for i in d1 if not i.startswith("__")]+[i for i in d if not i.startswith("__")]+[i for i in d1 if i.startswith("__")]+[i for i in d if  i.startswith("__")]

    def __getattr__(self,a):
        # print(a)
        if a in ["_instancecheck","papa","attr","prep"] or a.startswith('__'):
            return object.__getattribute__(self,a)
        if hasattr(self.get(),a):
            return getattr(self.get(),a)
        return super().__getattribute__(a)
    # def export(self,save=True,dirAdded=[],*args,**xargs):
    #     rep=super().export(save,dirAdded,*args,**xargs)
    #     rep['_prep']= copy.deepcopy(self.prep.dora.__dict__)
    #     return rep

    # @classmethod 
    # def _import(cls,loaded):
    #     rep = cls.__base__._import(loaded)
    #     # rep._prep = prep(rep,rep._prep)
    #     return rep

factoryCls.register_class(Datas)

from dora_study import Dora as Dora2
import types
from functools import wraps
class Dora(Dora2):
    def __init__(self, data = None, output = None):
        super().__init__(data,output)
        #for i in ["plot_feature","explore"]:
        #    delattr(self,i)
def saveLastDoraX(func,selfo,attr):
    @wraps(func)
    def with_logging(self,*args, **kwargs):
        type_=attr
        names= func.__name__
        fun2=getattr(selfo,names)
        # print(self,args,kwargs)
        # d=StudyClass(_data=getattr(self,i),_output=getattr(self,"target"))
        # d._data=getattr(self,type_)
        # d._output==getattr(self,"target")
        # kwargs["realFunc"]=realFunc
        # kwargs["realSelf"]=self
        kwargs["type_"]=[type_]
        print(self,args,kwargs)
        return fun2(*args,**kwargs)
    return with_logging
class DoraX:
    def _addmethod(self, name,method): 
        self.__dict__[name] = types.MethodType( method, self )

    def __init__(self, data = None, output = None,prep=None,attr=None):
        # super().__init__(data,output)
        self.data=data
        self.output=output
        self._prep=prep
        self._attr_=attr

        def use_snapshot(self, name):
            return self._prep.use_snapshot(name,type_=[attr])

        def back_initial_data(self):
            return self._prep.back_initial_data(type_=[attr])

        def back_one(self):
            return self._prep.back_one(type_=[attr])
            
        self._addmethod("use_snapshot",use_snapshot, )
        self._addmethod("back_initial_data",back_initial_data )
        self._addmethod("back_one", back_one )

        from dora_study import Dora
        fd=Dora.__dict__
        n=[i  for i,j in Dora.__dict__.items() if not i.startswith("_") and i not in ["plot_feature","explore"] and type(j)!=classmethod and hasattr(j,"__wrapped__")] 
        def job(g,i,wrapped=True):
            func=g.__wrapped__ if wrapped else g
            # a=get_args(func)
            # u=getVarInFn(a.signature)
            # uu=getNotVarInFn(a.signature)
            # o=uu+[f"type_=['{attr}']"]
            # fnu=make_fun(i,o+u)
            self._addmethod(i,saveLastDoraX(func,self._prep,attr))
        for i in n:
            job(fd[i],i)
        fd=Dora._CUSTOMS
        # print(fd)
        for i in fd:
            job(fd[i],i,False)
    
    
    def __getattr__(self,a_):
        # a=super().__getattr__(a_)
        a=a_
        if hasattr(self._prep,a) or (a.startswith("_") and not a.startswith("__")):
            # if hasattr(self._prep,"__wrapped__"):
            # print('d')
            return getattr(self._prep,a_)
        return super().__getattr__(a_)



class prepI:
    def __init__(self,l:Datas,dora=None):
        oname=l.y.name if hasattr(l.y,'name') else None
        if l.papa is None :#(l.papa is not None and l.papa._prep is None)
            self.dora=Dora(l.get(initial=True),output=oname) if dora is None else dora
        else:
            self.dora=DoraX(getattr(l.papa.prep,"train" if l.attr=='dataTrain' else "test"),output=oname,prep=l.papa.prep,attr="train" if l.attr=='dataTrain' else "test") if dora is None else dora
        self.attr=l.attr
        #self.speedML = l.
    def __dir__(self):
        return ["dora"]+[i for i in dir(self.dora) if hasattr(getattr(self.dora,i),"__wrapped__")]+[i for i in dir(self.dora) if not hasattr(getattr(self.dora,i),"__wrapped__")]

    def getData(self):
        if self.dora.__class__.__name__ == "Dora":
            return self.dora._data
        return getattr(self.dora,"train" if self.attr=='dataTrain' else "test")
   
    def __getattr__(self,b):
        # print("pb",b)
        if b in ["dora","_instancecheck","_ipython_display_"] or (b.startswith('__') or b.endswith('__')):
            return object.__getattribute__(self,b)
        return getattr(self.dora,b)

class DatasSupervise(Base):
    EXPORTABLE=["dataTrain","dataTest","_prep"]
    D=Datas

    def __init__(self,dataTrain:Datas=None,dataTest:Datas=None,_prep=None,ID=None):
        super().__init__(ID)
        self.dataTrain=dataTrain
        self.dataTest=dataTest
        self._prep=_prep
        # self.init()

    def init2(self,_prep=None):
        _prep=self._prep
        self._prep=_prep if _prep is not None else ( _prep if self.dataTrain is None else create_speedML(self))

    @classmethod
    def from_XY_Train_Test(cls,X_train,y_train,X_test,y_test,*,ID=None):
        return cls(cls.D(X_train,y_train),
                              cls.D(X_test,y_test),ID=ID)

    def get(self,deep=True,optsTrain={},optsTest={}):
        if deep:
            return [*self.dataTrain.get(concat=not deep,**optsTrain),*self.dataTest.get(concat=not deep,**optsTest)]
        return [self.dataTrain,self.dataTest]

    def __repr__(self,ind=2):
        txt=super().__repr__(ind=ind)
        nt="\n"+"\t"*ind
        stri=txt[:-1]+nt+"dataTrain : {},"+nt+"dataTest : {}]"
        return stri.format(securerRepr(self.dataTrain,ind+2),
                            securerRepr(self.dataTest,ind+2))


    @property
    def prep(self):
        if self._prep is None:
            self.init2()
        if self._prep is None:
            raise Exception("prep is not set")
        return self._prep

    @property
    def eda(self): 
        return self.prep.eda2()

factoryCls.register_class(DatasSupervise)
class Models(Base):
    EXPORTABLE=["models","namesModels","mappingNamesModelsInd","indNamesModels"]
    def __init__(self,models=None,names=None,indNames=None,mappingNames=None,ID=None):
        super().__init__(ID)
        self.models=models
        self.init(names=names,indNames=indNames,mappingNames=mappingNames)
        self.initModelsAndNames()
        
    def init(self,names=None,indNames=None,mappingNames=None):
        self.namesModels=names
        self.indNamesModels=indNames
        self.mappingNamesModelsInd=mappingNames

    def changeNames(self,x):
        self.init()
        self.namesModels=x
        self.initModelsAndNames()
        
    def initModelsAndNames(self):
        if self.models is not None:
            self.namesModels=self.namesModels if self.namesModels is not None else np.array(uniquify([getClassName(i) for i in self.models]))
            self.indNamesModels=self.indNamesModels if self.indNamesModels is not None else np.array(rangel(len(self.models)))
            self.mappingNamesModelsInd=self.mappingNamesModelsInd if self.mappingNamesModelsInd is not None else dict(zip(self.namesModels,self.indNamesModels))
            
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

factoryCls.register_class(Models)
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

            
factoryCls.register_class(Metric)
class CvOrigSorted(Base):
    EXPORTABLE=["original","sorted"]
    """docstring for CvOrigSorted"""
    def __init__(self, original=None, sorted=None, ID=None):
        super().__init__(ID)
        self.original = original
        self.sorted = sorted
        
factoryCls.register_class(CvOrigSorted)
class CvResultatsTrValOrigSorted(Base):
    """docstring for CvResultatsPreds"""
    EXPORTABLE=["Tr","Val"]
    def __init__(self,Tr:CvOrigSorted=None,Val:CvOrigSorted=None, ID=None):
        super().__init__(ID)
        self.Tr=Tr
        self.Val=Val
factoryCls.register_class(CvResultatsTrValOrigSorted)
class CvResultatsTrVal(Base):
    """docstring for CvResultatsPreds"""
    EXPORTABLE=["Tr","Val"]
    def __init__(self,Tr=None,Val=None, ID=None):
        super().__init__(ID)
        self.Tr=Tr
        self.Val=Val
factoryCls.register_class(CvResultatsTrVal)

class addPbToData():
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for train, test in tqdm(self.data):
            yield train, test

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
factoryCls.register_class(CvSplit)
class CvResultatsCvValidate(Base): pass
factoryCls.register_class(CvResultatsCvValidate)
class CvResultatsScores(CvResultatsTrVal): pass
factoryCls.register_class(CvResultatsScores)
class CvResultatsPreds(CvResultatsTrValOrigSorted): pass
factoryCls.register_class(CvResultatsPreds)
class CvResultatsDecFn(CvResultatsTrValOrigSorted): pass
factoryCls.register_class(CvResultatsDecFn)
from sklearn.metrics import classification_report, confusion_matrix
from ..viz import vizGet
class CvResultats(Base):
    EXPORTABLE=["preds","scores","cv_validate","decFn","name"]
    def __init__(self, preds:CvResultatsPreds=None, scores:CvResultatsScores=None, cv_validate=None, decFn=None, ID=None,name=None):
        super().__init__(ID)
        self.preds=preds
        self.scores=scores
        self.name=name
        self.cv_validate=cv_validate
        self.decFn=decFn



factoryCls.register_class(CvResultats)
# factoryCls.register_class(CvResultatsPreds)
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
        if self.resultats is not None:
            self.resultats = studyDico(self.resultats,papa=self,addPapaIf=lambda a:isinstance(a,Base),attr="resultats")

    def __repr__(self,ind=1):
        # txt="\n"
        txt=super().__repr__(ind=ind)
        nt="\n"+"\t"*ind
        stri=txt[:-1]+nt+"cv_ : {}"+nt+"resultats : {}"+nt+"args : [...]"
        #securerRepr(BeautifulDico(self.args),ind)
        return stri.format(securerRepr(self.cv,ind=ind+1),
            securerRepr(BeautifulDico(self.resultats),ind=ind+1))

    @classmethod 
    def import__(cls,ol,loaded,newIDS=False,*args,**xargs):
        rep=cls.import___(cls,ol,loaded,newIDS=newIDS,*args,**xargs)
        if rep.args["metric"] is not None and not isStr(rep.args["metric"]) and not isinstance(rep.args["metric"],Metric):
            rep.args["metric"]=Metric.import__(Metric(),rep.args["metric"])
        rep.resultats=studyDico(rep.resultats,papa=rep,addPapaIf=lambda a:isinstance(a,Base),attr="resultats")
        return rep

    @classmethod 
    def _import(cls,loaded):
        lo=super()._import(loaded)
        if isinstance(lo.resultats,dict):
            lo.resultats=studyDico(lo.resultats,papa=lo,addPapaIf=lambda a:isinstance(a,Base),attr="resultats")
        return lo

    @classmethod
    def fromCVItem(cls,cvItem):
        return cls(ID=cvItem.ID,cv=cvItem.cv,resultats=cvItem.resultats,args=cvItem.args)

#add std
    def resultatsSummary(self,roundVal=3,withStd=True):
        if not withStd:
            u=lambda i:(
                {k:getattr(v.scores,i) for k,v in self.resultats.items()}
                    | (pd.DataFrame |_funsInv_| dict(data=__.values(),
                                                    index=__.keys())).T \
                        | (np.round |_funsInv_| dict(a=__,decimals=roundVal)) \
                        | __.mean(axis=0).round(roundVal)\
                        #| _fun_.pd.concat(axis=1).T \
                        | __.to_frame().T | __.rename(index={0:i})
            )
        else:
            u=lambda i:(
                {k:getattr(v.scores,i) for k,v in self.resultats.items()}
                | (pd.DataFrame |_funsInv_| dict(data=__.values(),
                                                index=__.keys())).T \
                    | (np.round |_funsInv_| dict(a=__,decimals=roundVal)) \
                    | (listl |_funsInv_| [__.mean(axis=0).round(roundVal),__.std(axis=0).round(roundVal)])
                    |_fun_.pd.concat(axis=1).T \
                    | __.apply(lambda a:"{} ({})".format(a[0],a[1]),axis=0).to_frame().T | __.rename(index={0:i})
            )
        return u("Tr").append(u("Val"))

    def table_resultatsSummary(self,roundVal=3,title="Résultats crossValidés d'accuracy",
                                    marginT=50,width=450):
        s=self.resultatsSummary()
        s2=s.round(roundVal)\
            .table_plot()\
            .update_layout(title=title,
                           margin=dict(t=marginT),
                           width=width)
        return s2
factoryCls.register_class(CrossValidItem)
class CrossValid(Base):
    EXPORTABLE=["cv","parallel","random_state","shuffle","classifier","recreate","metric","models","nameCV","argu","namesMod"]
    def __init__(self,cv=None,classifier=None,metric:Metric=None,nameCV=None,parallel=True,random_state=42,
                 shuffle=True,recreate=False,models=None,namesMod=None,_models=None,_metric=None,
                 cviCls=CrossValidItem,cvrCls=CvResultats,papa=None):
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
            nameCV=self.nameCV,recreate=recreate,parallel=parallel,metric=_metric,models=_models)
        self.cv=CrossValid.getCV(self)
        self.cviCls=cviCls
        self.cvrCls=cvrCls
        self.papa_=papa
        
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
                nn=newStringUniqueInDico(names,dict(zip(names,[None]*len(names))))
                with ShowWarningsTmp():
                    warnings.warn("[CrossValid checkOK] name '{}' is already take, mtn c'est '{}'".format(self.nameCV,nn))
                self.nameCV=nn
        return True
            
    def computeCV(self,X,y,verbose=0,**xargs):
        if not isinstance(self.cv,CvSplit):
            self.cv = self.cv.split(X,y)
            self.cv=CvSplit.fromCvSplitted(self.cv)
        cv=self.cv
        cvo=self.crossV(X,y,verbose=verbose,**xargs)
        cl=self.cviCls
        ki=cl(self.nameCV,cv,cvo,self.argu)
        return (self.nameCV,ki)

    def crossV(self,X,y,verbose=0,n_jobs=-1,**xargs):
        cv=self.cv
        cvv=cv
        X=X
        if isinstance(X,pd.DataFrame):
            X=X.values
        metric=self.metric.scorer
        parallel=self.parallel
        models=self.models
        namesMod=[i.__class__.__name__ for i in models] if self.namesMod is None else self.namesMod
        #models=ifelse(models,models,self.models)
                           
        cvvTr=cvv.train
        cvvVal=cvv.validation
                           
        cvvTrCon=np.argsort(np.concatenate(cvvTr))
        cvvValCon=np.argsort(np.concatenate(cvvVal))
        cvvAll=cvv.all_
        cvvAll=addPbToData(cvvAll) if verbose >0 else cvvAll
        verbose = verbose if  verbose >1 else 0
        resu2=[cross_validate(mod ,X,y,return_train_score=True,
                            return_estimator=True,cv=cvvAll,n_jobs=ifelse(parallel,n_jobs),verbose=verbose,scoring=metric) for mod in models]

        preduVal=[[i.predict(X[k,:]) for i,k in zipl(resuI["estimator"],cvvVal) ] for resuI in resu2]
                           
        preduuVal=[np.concatenate(preduI)[cvvValCon] for preduI in preduVal]
        
        scoreVal = [resuI["test_score"] for resuI in resu2]
        
        preduTr=[[i.predict(X[k,:]) for i,k in zipl(resuI["estimator"],cvvTr) ] for resuI in resu2]
                           
        preduuTr=[np.concatenate(preduI)[cvvTrCon] for preduI in preduTr]
        
        scoreTr = [resuI["train_score"] for resuI in resu2]
        
        decVal=[[getDecFn(i,X[k,:]) for i,k in zipl(resuI["estimator"],cvvVal) ] for resuI in resu2]
        decVal2=[concatenateDecFn(preduI,cvvValCon) for preduI in decVal]

        resul={}
        for i,name in enumerate(namesMod):
            resul[name]=self.cvrCls(CvResultatsPreds(CvOrigSorted(preduTr[i],preduuTr[i]),
                                                      CvOrigSorted(preduVal[i],preduuVal[i])),
                                     CvResultatsScores(scoreTr[i],scoreVal[i]),
                                     resu2[i],
                                     CvResultatsDecFn(None,
                                                     CvOrigSorted(decVal[i],decVal2[i])),name=name )

        return resul
factoryCls.register_class(CrossValid)
from .hypertune import HyperTune
class BaseSupervise(Base):
    # @abstractmethod
    isClassif=False
    cvrCls=CvResultats
    cviCls=CrossValidItem
    EXPORTABLE=["datas","models","metric","cv","nameCvCurr","hypertune","pipeline"]
    EXPORTABLE_ARGS=dict(underscore=True)
    def __init__(self,ID=None,datas:DatasSupervise=None,
                    models:Models=None,metric:Metric=None,hypertune:HyperTune=None,
                    cv:Dict[str,CrossValidItem]=studyDico({}),nameCvCurr=None,pipeline=None,
                    *args,**xargs):
        super().__init__(ID)
        self._datas=datas
        self._models=models
        self._metric=metric
        self._cv=cv
        self._nameCvCurr=nameCvCurr
        self._hypertune=HyperTune() if hypertune is None else hypertune
        # self._isClassif=False
        self._pipeline=pipeline
        self.init()
    
    # @property
    # def isClassif(self):
    #     return self._isClassif
    
    # def __new__(cls,normal=True,*args,**xargs):
    #      return vizHelper(super(BaseSupervise,cls).__new__(cls,*args,**xargs))
    @property
    def vh(self):
        return vizHelper(self)
    
    @property
    def viz_(self):
        return self.vh.viz

    def init(self):
        # self._cv={}
        pass
        # self._nameCvCurr=None
        

    def setDataTrainTest(self,X_train=None,y_train=None,X_test=None,y_test=None,*,namesY=None,classif=False):
        #self.setDataX(X_train,X_test)
        D= str2Class('DatasSuperviseClassif') if classif else DatasSupervise
        self._datas=D.from_XY_Train_Test(X_train,y_train,
                                          X_test,y_test)

    def setModels(self,models,force=False,*args,**xargs):
        if self._models is not None and not force:
            with showWarningsTmp:
                warnings.warn("""
                        Models are alreary present in the study.
                        If you want to add models and computeCv -> addModelsToCurrCV(models)
                        If you want to force, set force=True (warnings CVs are based on this models)
                    """)
            return
        self._models=Models(models,*args,**xargs)

    def setMetric(self,metric):
        self._metric=Metric(metric)

    def computeCV(self,cv=5,random_state=42,shuffle=True,classifier=True,
                 nameCV=None,recreate=False,parallel=True,metric=None,
                 models=None,verbose=0,**xargs):
        _models=models
        _metric=metric
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
                 models=models,namesMod=namesMod,_models=_models,_metric=_metric,cviCls=self.cviCls,
                 cvrCls=self.cvrCls,papa=self)
        cv.checkOK(list(self.cv.keys()))
        resu=cv.computeCV(self.X_train,self.y_train,verbose=verbose,**xargs)
        self._cv[resu[0]]=resu[1]
        self._nameCvCurr=resu[0]
        
    
    @classmethod 
    def _import(cls,loaded):
        # print(loaded._cv)
        if isinstance(loaded._cv,dict):
            loaded._cv=studyDico(loaded._cv,papa=loaded,addPapaIf=lambda a:isinstance(a,Base),attr="cv")
        return loaded

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
        
    def getCurrCvResultats(self,i):
        return self.currCV.resultats[i]

    @property
    def hyperTune(self):
        return self._hypertune
    
    @property
    def currCV(self):
        return self._cv[self._nameCvCurr]
    
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
    def indNamesModels(self):
        return self._models.indNamesModels
    
    @property
    def mappingNamesModelsInd(self):
        return self._models.mappingNamesModelsInd
    
    @property
    def namesY(self):
        return self.train_datas.namesY

    @classmethod
    def getOrCreate(cls,id_,recreate=False,
                            clone=False,
                            deep=True,
                            cls_xargs=dict(),
                            imported=True,
                            *args,**xargs):
        # cls=self.__class__
        def clonee(rrt):
            return getStaticMethodFromObj(rrt,"clone")(rrt,deep=deep)
        def recreatee():
            rrt=cls(ID=id_,**cls_xargs)
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
            if imported:
                res = cls.Import(id_,*args,**xargs)
            else:
                res=cls.Load(id_,*args,**xargs)
            if res is None:
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

    def clone(self,ID=None,newIDS=False,deep=True,*args,**xargs):
        cl=super().clone(ID=ID,newIDS=newIDS,deep=deep,*args,**xargs)
        cl._cv=studyDico(cl.cv,papa=self,addPapaIf=lambda a:isinstance(a,Base),attr="cv")
        return cl

    def setPipeline(self,pipeline):
        self._pipeline=pipeline


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
 