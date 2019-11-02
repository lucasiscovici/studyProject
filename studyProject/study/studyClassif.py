from ..base import  BaseSupervise, Metric, DatasSupervise, Models, DatasSuperviseClassif
import os
from ..project import BaseSuperviseProject
from ..utils import getStaticMethodFromObj, getsourceP, getStaticMethodFromCls
from abc import ABC, abstractmethod
from interface import implements, Interface

class StudyClassif_:

    def computeCV(self,cv=3,random_state=42,shuffle=True,classifier=True,
                 nameCV=None,recreate=False,parallel=True,metric=Metric("accuracy"),
                 models=None,**xargs):
        return super().computeCV(cv,random_state,shuffle,classifier,nameCV,recreate,parallel,metric,models,**xargs)


class StudyClassif(StudyClassif_,BaseSupervise):
    EXPORTABLE=["datas"]
    EXPORTABLE_ARGS=dict(underscore=True)
    def __init__(self,
                 ID=None,
                 datas:DatasSuperviseClassif=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy")):
        super().__init__(
            ID=ID,
            datas=datas,
            models=models,
            metric=metric
        )
        self._datas=datas
    

class StudyClassifProject(StudyClassif_,BaseSuperviseProject):
    EXPORTABLE=["datas"]
    EXPORTABLE_ARGS=dict(underscore=True)
    def __init__(self,
                 ID=None,
                 datas:DatasSuperviseClassif=None,
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
        self._datas=datas
