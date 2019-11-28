from ..base import Base, DatasSupervise, BaseSupervise, factoryCls
from ..base.base import CrossValidItem, CvSplit, CvResultats, str2Class, getAnnotationInit, get_args_typing
from ..utils import getStaticMethodFromObj, ifelse, BeautifulDico, SaveLoad, mapl,takeInObjIfInArr, StudyDict, StudyClass, studyDico, getClassName, F
import os
from . import IProject
from typing import *
from interface import implements
from ..utils import isinstanceBase, isinstance, fnReturn
from ..viz import vizGet
# from ..study import DatasSuperviseClassif

class CrossValidItemProject(CrossValidItem):
    EXPORTABLE=["_based"]
    def __init__(self,ID:str=None,cv:CvSplit=None,resultats:Dict[str,CvResultats]={},args:Dict=None,*args_,**xargs):
        super().__init__(ID=ID,cv=cv,resultats=resultats,args=args,*args_,**xargs)
        self._based=None

    def __repr__(self,ind=1,*args,**xargs):
        nt="\n"+"\t"*ind
        rep=super().__repr__(ind=ind,*args,**xargs)
        return rep+nt+"_based: "+(self._based if self._based is not None else "None")

    #TODO
    # def changeID(self,id_):
        # self.ID=id_
    # @classmethod
    # def fromCVItem(cls,cvItem):
    #     return cls(ID=cvItem.ID,cv=cvItem.cv,resultats=cvItem.resultats,args=cvItem.args)

factoryCls.register_class(CrossValidItemProject)

class basedCv(Base):
    EXPORTABLE=["based","resu"]
    def __init__(self,based=None ,resu=None, ID=None):
        super().__init__(ID)
        self.based = based
        self.resu = resu


    @staticmethod
    def Export___(cls,obj,save=True,version=None,papaExport=[],*args,**xargs):
        # print("ici")# TODO: TWO LOOP ON wHY ?
        expo=obj.resu
        bb=obj.based
        if bb is None:
            expo=expo.export(save=False,*args,**xargs)
        bb = "None" if bb is None else bb
        repu=dict(based=bb,resu=expo)
        repu["____cls"]=cls.__name__
        return repu

    @classmethod 
    def import__(cls,ol,loaded,me="basedCv",*args,**xargs):
        # print("ici")# TODO: TWO LOOP ON wHY ?
        # print( cls.__name__ )
        # print( ol.__dict__ )
        # print(loaded)
        # if not isinstance(ol,cls):
        bas=loaded["based"]
        resu=loaded["resu"]
        if bas == "None" or bas is None:
            # print("ici")
            cl2o=resu["____cls"]
            cl2=str2Class(cl2o)
            resu=cl2.import__(cl2(),resu)
            bas=None

        rep=cls(bas,resu)
        # print(rep)
        # print(rep.__dict__)
        return rep
        # return ol

factoryCls.register_class(basedCv)
from collections import defaultdict
class StudyProject(Base):
    # DEFAULT_REP="study_project"
    # DEFAULT_EXT=".studyProject"
    EXPORTABLE=["studies","curr","data","cv","cvOpti","dataOpti"]
    EXPORTABLE_ARGS=dict(underscore=True)
    def __init__(self,ID=None,studies:Dict[str,BaseSupervise]=None,
                    curr=None,data:Dict[str,DatasSupervise]=None,cv:Dict[str,basedCv]={},
                    cvOpti=True,dataOpti=True):
        super().__init__(ID)
        self._studies={} if studies is None else studies
        self._curr=curr
        self._data=data if data is not None else {} #BeautifulDico({"_ZERO":DatasSupervise.from_XY_Train_Test(None,None,None,None,None,ID="_ZERO")})
        self._cv={}
        self._cvOpti=cvOpti
        self._dataOpti=dataOpti
        # self._cvK=defaultdict(list);

    @property
    def vh(self):
        return vizHelper(self)

    def addCV(self,name,cv):
        if self._cvOpti:
            if cv._based is not None:
                rpo=list(self._cv[cv._based].resu.resultats.keys())
                self._cv[cv._based]=basedCv(name,rpo)
                # self._cvK[name]=se .lf._cvK[name]+[cv._based]
            self._cv[name]=basedCv(None,cv)            
        

    def add(self,study_):
        study_2=study_
        if isinstance(study_2,implements(IProject)):
            study_2._project=self
        self._studies[study_.ID]=study_2
        self._curr=study_2.ID
        return study_2
    
    def get(self,id_=None):
        id_=ifelse(id_,id_,self._curr)
        return self._studies[id_]
    
    def getAndCurr(self,id_):
        self._curr=id_
        return self.get(od_)
        
    def addOrGetStudy(self,id_,cls,recreate=False,clone=False,deep=True,vh=True,import_kwargs=dict(),imported=False):
        def clonee(rrt):
            return getStaticMethodFromObj(rrt,"clone")(rrt,deep=deep)
        def recreatee():
            rrt=cls(ID=id_)
            if clone :
                rrt=clonee(rrt)
            res =self.add(rrt)
            return res
        def cloneStudy():
            ru=self._studies[id_] 
            ru=clonee(ru)
            self._studies[id_]=ru
            return self._studies[id_]
        if recreate:
            res=recreatee()
        else:
            if imported:
                res= cls.Import(id_,**import_kwargs)
                res = clonee(res) if clone else res
                self.add(res)
            else:
                res=ifelse(id_ in self._studies,
                          lambda:self._studies[id_] if not clone else cloneStudy() ,
                          lambda:recreatee())()
        if isinstance(res,implements(IProject)):res.check() 
        self._curr=id_
        # if vh:
        # res=res.vh
        return res
    
    @property
    def currStudy(self):
        return self.get()
        
    
    @classmethod
    def getOrCreate(cls,ID,repertoire=None,ext=None,
                    path=os.getcwd(),delim="/",imported=True,noDefaults=False,
                    recreate=False,clone=False,deep=True,chut=True,save_load_load={},save_load_get_path={},import_kwargs={}):
        from . import IProject
        # repertoire = ifelse(repertoire is None,StudyProject.DEFAULT_REP,repertoire)
        # ext=ifelse(ext is None,StudyProject.DEFAULT_EXT,ext)
        def clonee(rrt):
            return getStaticMethodFromObj(rrt,"clone")(rrt,deep=deep)
        # repo=path+delim+repertoire
        # print(repo)
        if recreate:
            return StudyProject(ID)
        # if imported:

        # repo = cls.get_repertoire(repertoire)
        repertoire,ext=cls.get_rep_ext(repertoire,ext,chut=chut)

        # repos=cls.build_repertoire(repertoire,path=os.getcwd(),dp=cls.DEFAULT_PATH if not noDefaults else "",delim="/",
                            # fn=lambda: StudyProject(ID),returnFn=True)
        repos,(ok,filo)=cls.build_rep_ext(repertoire,ext,ID,dp=cls.DEFAULT_PATH if not noDefaults else "",
                                            chut=chut,recreate=recreate,returnFn=True,fn=lambda repo: StudyProject(ID))
        # print(repos)
        # print(ok)
        # print(filo)
        # raise Exception("kd")
        if len(repos)>1:
            return repos[1]
        # print(repo)
        # if not os.path.exists(repo):
        #     return StudyProject(ID)

        # filo,ok=cls.build_ext(repo,cls.get_ext(ext),ID,
            # delim=delim,recreate=recreate,chut=chut)
        # print(filo)
        filo=SaveLoad.getPath(filo,addExtension=True,**save_load_get_path)
        if not os.path.isfile(filo):
            return StudyProject(ID)

        if imported:
            return cls.Import(filo,addExtension=False,chut=chut,noDefaults=True,path="",**import_kwargs)
        
        sl=SaveLoad.load(filo,addExtension=False,chut=chut,**save_load_load)
        # sl=cls.Load(filo,noDefaults=True,addExtension=False,path="",**xargs)
        sf={}
        for k,v_ in sl._studies.items():
            v=ifelse(clone,lambda: clonee(v_),lambda:v_)()
            if isinstance(v,implements(IProject)):
                v.begin()
                v.setProject(sl)
                #print(v.idData)
                if sl.dataOpti and v.getProprocessDataFromProjectFn() is not None:
                    v.setDataTrainTest(id_=v.getIdData(),force=True)
                    try:
                        v.proprocessDataFromProject(v.getProprocessDataFromProjectFn(),**v._proprocessDataFromProjectFnOpts)
                    except Exception as e:
                        raise e
                        warnings.warn("[StudyProject getOrCreate] pb with {} when proprocessDataFromProject".format(k))
                        pass
                        #print("Error")
                        #print(inst)
                            #print(v.isProcess)
                v.check()
            sf[k]=v
        sl._studies=sf
        return sl
    

    def cvM(self,v_):
        f=v.cv_
        # for k,v in f.items():


    def save(self,repertoire=None,ext=None,path=os.getcwd(),
             delim="/",returnOK=False,**xargs):
        ID=self.ID
        cvOpti=self._cvOpti
        dataOpti=self._dataOpti
        # repertoire = ifelse(repertoire is None,StudyProject.DEFAULT_REP,repertoire)
        # ext=ifelse(ext is None,StudyProject.DEFAULT_EXT,ext)
        # repo=path+delim+repertoire
        # if not os.path.exists(repo):
        #     os.makedirs(repo)
        # filo=repo+delim+ID+ext
        # sl=StudyProject.clone(self,deep=False)
        # sl=StudyProject.clone(self,deep=True)
        ff={}
        for k,v_ in self._studies.items():
            v=v_.clone(self)
            # print(isinstance(v_,implements(IProject)))
            if isinstance(v_,implements(IProject)):
                # v__=v.project
                # v=v_.clone(withoutProject=True)
                # li2=v._idCvBased
                li=v.getIdData()
                # v._project=v__
                if dataOpti and v.getProprocessDataFromProjectFn() is not None:
                    v._datas={}
                #v._cv={}#
                #\____/=\____/#
                if cvOpti:
                    ddd=list(vizGet(v._cv.keys())) if isinstance(v._cv,dict) else v._cv
                    # for k2,v2 in v._cv.items():
                    v._cv=ddd
                # v.setDataTrainTest(id_="_ZERO")
                # v.setIdData(li)
                v.setProject(v.ID)

                # print(v)
            ff[k]=vizGet(v)

        # print(ff)
        stu=self._studies
        cvv=self._cv
        self._studies=ff
        # copye=StudyProject.clone(sl,deep=True)
        # copye._studies=ff
        # sl=copye
        def reloadS(self,stu,cvv):
            self._studies=stu
            self._cv=cvv
        if returnOK:
            return StudyClass(obj=self,fin=lambda obj=obj,stu=stu,cvv=cvv: reloadS(obj,stu,cvv))
        else:
            self.__class__.Save(self,ID,repertoire=repertoire,ext=ext,path=path,delim=delim,**xargs)
            self._studies=stu
            self._cv=cvv
            # SaveLoad.save(sl,filo,**xargs)
    
    
    
    def clone_(self,ID=None,deep=False):
        return getStaticMethodFromObj(self,"clone")(ID,deep)
        
    # @staticmethod
    # def clone(self,ID=None,deep=False):

    #     cloned=Base.clone(self,ID=ID,deep=deep)
    #     if deep:
    #         cloned._studies={k:getStaticMethodFromObj(v,"clone")(v,v.ID,deep=True) for k,v in self._studies.items()}
    #     return cloned
    
    @staticmethod
    def fromStudyProject(self,studyG):
        return studyG.clone(studyG.name)
    
    def saveDatasWithId(self,id_,X_train,y_train,X_test,y_test,namesY=None,classif=False):
        D=str2Class("DatasSuperviseClassif") if classif else DatasSupervise

        self._data[id_]=D.from_XY_Train_Test(*[X_train,y_train,X_test,y_test],ID=id_)

    def saveDataSupervise(self,dataSup):
        id_= dataSup.ID
        self._data[id_] =  dataSup.clone()

    @staticmethod
    def plan(fromHelp=False):
        print("PLAN Project:")
        if not fromHelp:
            print("\tCREATE Study Project:")
            print("\t\t[Name_Variable_Study_Project].getOrCreate(ID_OF_PROJECT)")
        print("\tSET Project Data (when original data are the same for multiple study):")
        print("\t\tWith X,y (train and test) and labels of y:"+"\n\t\t\t[Name_Variable_Study_Project].saveDatasWithId(ID_OF_DATA,X_train, y_train, X_test, y_test, namesY)")
        print("\tCreate or Get study inside the project:")
        print("\t\t[Name_Variable_Study_Project].addOrGetStudy(ID_STUDY,CLASS_OF_STUDY__OR__OTHER_STUDY_CLONE)")
        print("\t\t\tEx: [Name_Variable_Study_Project].addOrGetStudy('HELLO_WORLD',StudyClassifProject) ")
        print("\t\t\tEx: [Name_Variable_Study_Project].addOrGetStudy('HELLO_WORLD',[Name_Variable_Study].clone)")
        print("\tSAVE:")
        print("\t\t[Name_Variable_Study].save()")

    def help(self):
        getStaticMethodFromObj(self,"plan")(fromHelp=True)

    def __repr__(self,ind=1,onlyID=False):
        t="\t"*ind
        nt="\n"+t
        ntt=""
        ff=self._studies
        if onlyID:
            return ("[[StudyProject]"+nt+"ID : {}]").format(self.ID)
        if len(ff)>0:
            ntt=nt+"\t"+t.join(mapl(lambda a:"ID : "+str(a),ff))
            # print(ntt)
            rep= ("[[StudyProject]"+nt+"ID : {}"+nt+"Studies : {}]").format(self.ID,ntt)
        else:
            ntt="0"
            rep =("[[StudyProject]"+nt+"ID : {}"+nt+"Studies : {}]").format(self.ID,ntt)
        if len(self._data)>0:
            ntt=nt+"\t"+(nt+"\t").join(mapl(lambda a:"ID : "+str(a),self._data))
            rep= (rep[:-1]+nt+"Datas : {}]").format(ntt)
        else:
            ntt="0"
            rep =(rep[:-1]+nt+"Datas : {}]").format(ntt)
        return rep

    def export(self,save=True,*args,**xargs):
        objClss=self.save(returnOK=True)
        if not self._cvOpti:
            objClss.obj._cv={}
        rep=self.__class__.Export(objClss.obj,save=save,*args,**xargs)
        objClss.fin()
        return rep

    @staticmethod
    def import_give_me_cv(proj,cvID,clsStudy=None,maxou=50):
        dd=proj.cv
        me=dd[cvID]
        meb=me.based
        meresu=me.resu
        # print("ddd",me,",",meb,",",meresu)
        if meb is None:
            if clsStudy is not None and  ( clsStudy.__name__ != meresu.__class__.__name__):
                resul=clsStudy.import__(clsStudy(),
                                          meresu.export(save=F))
                resul2=resul.resultats
                resul.resultats=studyDico(resul2)
                meresu=resul
            else:
                meresu.resultats=studyDico(meresu.resultats)
            meresu=meresu._import(meresu)
            return meresu

        i=0
        while meb is not None:
            me2=dd[meb]
            meb=me2.based
            i=i+1
            if i==maxou:
                raise Exception("PB import_give_me_cv")
        resul=me2.resu.clone()
        # print(clsStudy,resul.__class__.__name__)
        if clsStudy is not None and  ( clsStudy.__name__ != resul.__class__.__name__):
            resul=clsStudy.import__(clsStudy(),
                                      resul.export(save=F))
        resul2={}
        for i in meresu:
            resul2[i]=resul.resultats[i]
        resul.resultats=studyDico(resul2)
        resul=resul._import(resul)
        return resul




    @classmethod 
    def _import(cls,loaded,clone=False):
        sf={}
        sl=loaded
        # print(sl)
        if sl is None:
            return sl
        for k,v_ in sl._studies.items():
            v=ifelse(clone,lambda: clonee(v_),lambda:v_)()
            # print(v)
            if isinstance(v,implements(IProject)):
                # print(v_)
                v.begin()
                v.setProject(sl)
                if sl.dataOpti and v.getProprocessDataFromProjectFn() is not None: #TODO: save function getProprocessDataFromProjectFn when different python version
                    # print(v.getIdData())
                    v.setDataTrainTest(id_=v.getIdData(),force=True)
                    try:
                        v.proprocessDataFromProject(v.getProprocessDataFromProjectFn(),**v._proprocessDataFromProjectFnOpts)
                    except Exception as e:
                        raise e
                        warnings.warn("[StudyProject getOrCreate] pb with {} when proprocessDataFromProject".format(k))
                        #print("Error")
                        #print(inst)
                            #print(v.isProcess)preproces_binary
                if sl._cvOpti:
                    clsCV= getAnnotationInit(v)
                    clsCV= get_args_typing(clsCV["cv"])[1] if "cv" in clsCV else None
                    v._cv  = studyDico({k:cls.import_give_me_cv(sl,k,clsStudy=clsCV) for k in v._cv})
                v.check()
                v=v._import(v)
            sf[k]=v
        sl._studies=studyDico(sf)
        return sl

    # @classmethod
    # def _export(cls,obj):
    #     obj=obj.save(returnOK=True)
    #     # rep=takeInObjIfInArr(cls.EXPORTABLE,obj)

    #     # rep={k:(v.export(save=False) if isinstance(v,Base) else v) for k,v in rep.items()}
    #     # rep["data"] = {k:v.export(save=False) for k,v in rep["data"].items()}
    #     # rep["studies"] = {k:v.export(save=False) for k,v in rep["studies"].items()}
    #     return obj


from ..base import *
from ..base.base import CrossValidItem, CvSplit, CvResultats
import numpy as np
import copy
import os
import warnings
from .interfaceProject import IProject
from abc import abstractmethod
from interface import implements, Interface
from ..utils import securerRepr, mapl , isStr, randomString
from typing import List, Dict
from studyPipe.pipes import *
    
class BaseSuperviseProject(BaseSupervise,implements(IProject)):
    EXPORTABLE=["project","idDataProject","proprocessDataFromProjectFn",
    "proprocessDataFromProjectFnOpts","isProcessedDataFromProject","cv"]
    EXPORTABLE_ARGS=dict(underscore=True)
    cvrCls=CvResultats
    cviCls=CrossValidItemProject
    @abstractmethod
    def __init__(self,ID=None,datas:DatasSupervise=None,
                        models:Models=None,metric:Metric=None,cv:Dict[str,CrossValidItemProject]=studyDico({}),
                        project:StudyProject=None,*args,**xargs):
        super().__init__(ID,datas,models,metric)
        self._project=project
        self._cv=cv

    def init(self):
        super().init()
        self._idDataProject=None
        # self._idCvBased=None
        # self._cv=StudyDict()
        self._proprocessDataFromProjectFn=None
        self._proprocessDataFromProjectFnOpts={}
        self.begin()

    def begin(self):
        self._isProcessedDataFromProject=False

    def setProject(self,p):
        self._project=p

    def getProject(self):
        return self.project

    def getProprocessDataFromProjectFn(self):
        return self.proprocessDataFromProjectFn

    def getIdData(self):
        return self._idDataProject

    def setIdData(self,i):
        self._idDataProject=i

    def setDataTrainTest(self,X_train=None,y_train=None,
                              X_test=None,y_test=None,
                              namesY=None,id_=None,force=False):

        if self.isProcessedDataFromProject and not force:
            raise Exception("[BaseSuperviseProject setDataTrainTest] processing deja fait pour les données du projet (et force est à False)")

        if id_ is None and np.any(mapl(lambda a:a is None,[X_train,X_test,y_train,y_test])):
           raise KeyError("if id_ is None, all of [X_train,X_test,y_train,y_test] must be specified  ")
        if id_ is not None and self.project is None:
            raise KeyError("if id_ is specified, project must be set")
        if id_ is not None and id_ not in self.project.data:
            raise KeyError("id_ not in global")
        if id_ is not None:
            # y=self.project.data[id_]
            super().setDataTrainTest(*self.project.data[id_].get())
            # self._datas=self.project.data[id_]
            self._idDataProject=id_
        else:
            classif = self.isClassif
            super().setDataTrainTest(X_train,y_train,X_test,y_test,classif=classif)

    def proprocessDataFromProject(self,fn=None,force=False,pipelineX=None, pipelineY=None):
        classif = self.isClassif
        if self.isProcessedDataFromProject and not force:
            raise Exception("[BaseSuperviseProject proprocessDataFromProject] processing deja fait pour les données du projet (et force est à False)")
            
        if fn is not None:
            self._proprocessDataFromProjectFn = fn
            # self._proprocessDataFromProjectFnOpts=dict(classif=classif)
            super().setDataTrainTest(*fn(*self._datas.get(deep=True,optsTrain=dict(withNamesY=False))),classif=classif)
            self._isProcessedDataFromProject = True
            if pipelineX is not None:
                self.pipelineX=pipelineX
            if pipelineY is not None:
                self.pipelineY=pipelineY
        elif fn is None:
            X_train,y_train,X_test,y_test=self.datas.get()
            if pipelineX is not None:
                X_train=pipelineX.fit_transform(X_train)
                X_test=pipelineX.fit_transform(X_test)
            if pipelineY is not None:
                y_train=pipelineY.fit_transform(y_train)
                y_test=pipelineY.fit_transform(y_test)
            return self.proprocessDataFromProject(fnReturn((X_train,y_train,X_test,y_test)),pipelineX=pipelineX,pipelineY=pipelineY,force=force)

    def check(self):
         if not self.isProcessedDataFromProject and self.proprocessDataFromProjectFn is not None:
            warnings.warn("Attention vous devez appeler impérativement  la méthode proprocessDataFromProject de l'object '{}' reçu pour que les données soit les bonnes".format(getClassName(self)))

    def __repr__(self,ind=1,orig=False):
        if orig:
            return object.__repr__(self)
        txt=super().__repr__(ind=ind)
        nt="\n"+"\t"*ind
        stri=txt[:-1]+nt+"project : {},"+nt+"idDataProject : {},"+nt+"proprocessDataFromProjectFn : {},"+nt+"isProcessedDataFromProject : {}]"
        # print(securerRepr(self.project,ind+2,onlyID=True))
        # print(self)
        # print(stri)
        return stri.format(securerRepr(self.project,ind+2,onlyID=True),self.idDataProject,self.proprocessDataFromProjectFn,self.isProcessedDataFromProject)

    def clone(self,ID=None,withoutProjects=True,newIDS=False,*args,**xargs):
        p=self._project
        self._project=(p.ID if not isStr(p) else p) if p is not None else None
        r=super().clone(ID=ID,newIDS=newIDS,*args,**xargs)
        self._project=p
        r._project=p
        return r

    @classmethod
    def Export(cls,obj,save=True,saveArgs={},me="BaseSuperviseProject",*args,**xargs):
        # print("ici")# TODO: TWO LOOP ON wHY ?
        po=obj._project
        if po is not None and not isStr(po):
            obj._project=po.ID
        oo=cls.Export__(cls,obj,save=save,saveArgs=saveArgs,*args,**xargs)
        
        try:
            oo._project=po
        except: 
            pass
        return oo

    @classmethod 
    def import__(cls,ol,loaded,me="BaseSuperviseProject",*args,**xargs):
        # print("ici")# TODO: TWO LOOP ON wHY ?
        # print(loaded)
        if loaded is None:
            return cls.import___(cls,ol,loaded,*args,**xargs)
        # print("p",loaded["ID"])
        po=loaded["_project"]
        if po is not None and isStr(po):
            loaded["_project"]=None
        # print(loaded["_project"])
        rep=cls.import___(cls,ol,loaded,*args,**xargs)
        # if isStr(po):
        rep._project= po
        return rep


    def addModelsToCurrCV(self,models:List,names=None ,nameCV=None,*args,**xargs):
        cvCurr=self.currCV
        li=cvCurr.ID
        cloneE=self.clone()
        cloneE.setModels(self.models+models,force=True)
        # return cloneE
        namesM=cloneE.namesModels[-len(models):]
        cloneE.setModels(models,names=namesM if names is None else names,force=True)
        params=cvCurr.args
        params["cv"]=cvCurr.cv
        # params["nameCV"]=randomString()
        if "namesMod" in params: del params["namesMod"]
        params["noAddCv"]=True
        params["recreate"]=True
        cloneE.computeCV(*args,**params,**xargs)
        res=cloneE.currCV.resultats
        cvCurr.ID=randomString() if nameCV is None else nameCV
        for k,v in res.items():
            cvCurr.resultats[k]=v
        cvCurr._based = li
        self.setModels(self.models+models,force=True)
        self._nameCvCurr=cvCurr.ID
        self._cv[cvCurr.ID]=self._cv[li]
        del self._cv[li]
        # self.idCvBased=li
        self._project.addCV(cvCurr.ID,self._cv[cvCurr.ID])



    def computeCV(self,cv=5,random_state=42,shuffle=True,classifier=True,
                 nameCV=None,recreate=False,parallel=True,metric=None,
                 models=None,noAddCv=False,**xargs):
        rep=super().computeCV(cv=cv,random_state=random_state,shuffle=shuffle,classifier=classifier,
                 nameCV=nameCV,recreate=recreate,parallel=parallel,metric=metric,
                 models=models,**xargs)
        # print(rep)
        classif=self.isClassif
        D= self.cviCls
        self._cv[self._nameCvCurr]=D.fromCVItem(self.currCV)
        # self._nameCvCurr=resu[0]
        # rep._based = None
        if not noAddCv:self._project.addCV(self._nameCvCurr,self.currCV)

        return rep

# from ..study.studyClassif import CvResultatsClassif
# from ..study.studyClassif import CrossValidItemClassifProject
# from ..study.studyClassif import DatasSuperviseClassif