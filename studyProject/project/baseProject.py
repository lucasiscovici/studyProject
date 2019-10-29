from .project import StudyProject
from ..base import *
import numpy as np
import copy
import os
import warnings
from .interfaceProject import IProject
from abc import abstractmethod
from interface import implements, Interface
from ..utils import securerRepr, mapl , isStr

class BaseSuperviseProject(BaseSupervise,implements(IProject)):
    EXPORTABLE=["project","idDataProject","proprocessDataFromProjectFn",
    "isProcessedDataFromProject"]
    EXPORTABLE_ARGS=dict(underscore=True)
    
    @abstractmethod
    def __init__(self,ID=None,datas:DatasSupervise=None,
                        models:Models=None,metric:Metric=None,project:StudyProject=None,*args,**xargs):
        super().__init__(ID,datas,models,metric)
        self._project=project

    def init(self):
        super().init()
        self._idDataProject=None
        self._proprocessDataFromProjectFn=None
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
                              namesY=None,id_=None):
        if id_ is None and np.any(mapl(lambda a:a is None,[X_train,X_test,y_train,y_test])):
           raise KeyError("if id_ is None, all of [X_train,X_test,y_train,y_test] must be specified  ")
        if id_ is not None and self.project is None:
            raise KeyError("if id_ is specified, project must be set")
        if id_ is not None and id_ not in self.project.data:
            raise KeyError("id_ not in global")
        if id_ is not None:
            self._datas=self.project.data[id_]
            self._idDataProject=id_
        else:
            super().setDataTrainTest(X_train,y_train,X_test,y_test,namesY)

    def proprocessDataFromProject(self,fn=None,force=False):
        if self.isProcessedDataFromProject and not force:
            warnings.warn("[BaseSuperviseProject proprocessDataFromProject] processing deja fait pour les données du projet (et force est à False)")
        if fn is not None:
            self._proprocessDataFromProjectFn = fn
            super().setDataTrainTest(*fn(*self._datas.get(deep=True,optsTrain=dict(withNamesY=False))))
            self._isProcessedDataFromProject = True

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

    def clone(self,withoutProject=True,*args,**xargs):
        p=self.project
        self._project=p.ID if p is not None else None
        r=super().clone(*args,**xargs)
        self._project=p
        return r

    @classmethod
    def Export(cls,obj,save=True,saveArgs={},me="BaseSuperviseProject"):
        # print("ici")# TODO: TWO LOOP ON wHY ?
        po=obj._project
        if not isStr(po):
            obj._project=po.ID
        return cls.Export__(cls,obj,save=save,saveArgs=saveArgs)

    @classmethod 
    def import__(cls,ol,loaded,me="BaseSuperviseProject"):
        # print("ici")
        # print(loaded)
        if loaded is None:
            return cls.import___(cls,ol,loaded)
        # print("p",loaded["ID"])
        po=loaded["_project"]
        if isStr(po):
            loaded["_project"]=None
        # print(loaded["_project"])
        rep=cls.import___(cls,ol,loaded)
        if po is str:
            rep["_project"]= po
        return rep



    

