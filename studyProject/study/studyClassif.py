from ..base import  BaseSupervise, Metric, DatasSupervise, Models
import os
from ..project import BaseSuperviseProject
from ..utils import getStaticMethodFromObj, getsourceP, getStaticMethodFromCls
from abc import ABC, abstractmethod
from interface import implements, Interface

class StudyClassif_:

    def computeCV(self,cv=3,random_state=42,shuffle=True,classifier=True,
                 nameCV=None,recreate=False,parallel=True,metric=Metric("accuracy"),
                 models=None):
        # print("la")
        return super().computeCV(cv,random_state,shuffle,classifier,nameCV,recreate,parallel,metric,models)

    # def save(self,repertoire=None,ext=None,ID=None,path=os.getcwd(),
    #          delim="/",recreate=False,**xargs):
    #     # print(recreate)
    #     return super().save(repertoire if repertoire is not None else getStaticMethodFromObj(self,"DEFAULT_REP"),
    #                     ext if ext is not None else getStaticMethodFromObj(self,"DEFAULT_EXT"),
    #                     ID,path,delim,recreate,**xargs)

    # @staticmethod
    # def Save(self,
    #          ID=None,
    #          repertoire=None,
    #          ext=None,
    #          path=os.getcwd(),
    #          delim="/",
    #          recreate=False,
    #          **xargs):
    #     # print("ici")
    #     return self.save(repertoire if repertoire is not None else getStaticMethodFromObj(self,"DEFAULT_REP"),
    #                     ext if ext is not None else getStaticMethodFromObj(self,"DEFAULT_EXT"),ID,path,delim,recreate,**xargs)

    # @classmethod
    # def Load(cls,ID,
    #          repertoire=None,
    #          ext=None,
    #          path=os.getcwd(),
    #          delim="/",
    #         **xargs):

    #     return cls.__bases__[-1].Load(ID,repertoire if repertoire is not None else getStaticMethodFromCls(cls,"DEFAULT_REP"),
    #                                 ext if ext is not None else getStaticMethodFromCls(cls,"DEFAULT_EXT"),
    #                                 path,delim,**xargs)


class StudyClassif(StudyClassif_,BaseSupervise):
    def __init__(self,
                 ID=None,
                 datas:DatasSupervise=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy")):
        super().__init__(
            ID=ID,
            datas=datas,
            models=models,
            metric=metric
        )
    

class StudyClassifProject(StudyClassif_,BaseSuperviseProject):
    def __init__(self,
                 ID=None,
                 datas:DatasSupervise=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy"),
                 project=None): 
        super().__init__(
            ID=ID,
            datas=datas,
            models=models,
            metric=metric,
            project=project
        )
