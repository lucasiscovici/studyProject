from ..base import  BaseSupervise, Metric, DatasSupervise, Models, CrossValidItem
import os
from ..project import BaseSuperviseProject
from ..utils import getStaticMethodFromObj, getsourceP, getStaticMethodFromCls, isNumpyArr, listl, zipl
from abc import ABC, abstractmethod
from interface import implements, Interface
from ..base import DatasSupervise, Datas, factoryCls
import warnings

import warnings as warning
import pandas as pd
import numpy as np
from studyPipe.pipes import * 
from ..viz.viz import vizHelper
from ..utils import isinstanceBase, isinstance
from typing import Dict
class DatasClassif(Datas):
    EXPORTABLE=["cat"]
    EXPORTABLE_ARGS=dict(underscore=False)
    # y must be a series or a dataframe

    def __init__(self,X=None,y=None,cat=None,ID=None):
        super().__init__(X,y,ID)
        self.cat=cat
        self.init()

    def init(self):
        if self.y is not None:
            if isNumpyArr(self.y):
                if np.ndim(self.y)>1:
                    raise Exception("PB")
                self.y=pd.Series(self.y)
            if not isinstance(self.y,pd.Series) or isinstance(self.y,pd.DataFrame):
                raise Exception("y must be a pd.Series or pd.DataFrame")

            if isinstance(self.y,pd.DataFrame):
                l={i:j.name for i,j in self.y.dtypes.values.to_dict().items()}
                ll={i:(j!="categorical") for i,j in l.items()} 
                if sum(ll.values())>0:
                    warnings.warn(
                        "{} col not are not categorical, must be"
                    )
                    for i in ll.keys():
                        if ll[i]:
                            self.y[i]=self.y[i].astype('category')
                cat={}
                for i in ll.keys():
                    cat[i]=self.y[i].cat.categories.tolist()
                self.cat=cat
            elif isinstance(self.y,pd.Series):
                if self.y.dtype.name != "category":
                    warning.warn(
                        "self.y must be categorical"
                        )
                self.y=self.y.astype("category")
            self.cat=self.y.cat.categories.tolist()

    def class_balance(self,normalize=True):
        df=self.y.value_counts(normalize=normalize).set_name("Class Balance")
        return df

factoryCls.register_class(DatasClassif)

class DatasSuperviseClassif(DatasSupervise):
    EXPORTABLE=["dataTrain","dataTest"]
    D=DatasClassif
    def __init__(self,dataTrain:DatasClassif=None,dataTest:DatasClassif=None,ID=None):
        super().__init__(ID=ID)
        self.dataTrain=dataTrain
        self.dataTest=dataTest


    def class_balance(self,normalize=False):
        rep=(
            self 
            | ((__.dataTrain,__.dataTest) |_funs_| listl )
            | ((__,["dataTrain","dataTest"]) |_funs_| zipl )
            | _ftools_.mapl(__[0].class_balance(normalize=normalize).to_frame(__[1]))
            | (pd.concat |_funsInv_| dict(objs=__,axis=1))
        )
        return rep
factoryCls.register_class(DatasSuperviseClassif)
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
                 metric:Metric=Metric("accuracy"),
                 cv:Dict[str,CrossValidItem]=None,
                 nameCvCurr=None,dejaINIT=False,normal=True):
        if not dejaINIT:
            if cv is None:
                cv={}
            super(StudyClassif,self).__init__(
                ID=ID,
                datas=datas,
                models=models,
                metric=metric,
                cv=cv,
                nameCvCurr=nameCvCurr
            )
            self._datas=datas
        # print('cv',self._cv)
    def __new__(cls,
                 ID=None,
                 datas:DatasSuperviseClassif=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy"),
                 cv:Dict[str,CrossValidItem]=None,
                 nameCvCurr=None,
                 normal=True):
        # print(cv) 
        instance= super(StudyClassif,cls).__new__(cls)
        # print(instance)
        if not normal:
            # print(cv)
            instance.__init__(ID=ID,
                                    datas=datas,
                                    models=models,
                                    metric=metric,
                                    cv=cv,
                                    nameCvCurr=nameCvCurr)
            # print(instance)
        # print(instance)
        # print('cv',instance._cv)
        return instance.vh if not normal else instance

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
        # print('cv',self.cv)

    def __new__(cls,
                 ID=None,
                 datas:DatasSuperviseClassif=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy"),
                 project=None,normal=True):
        instance= super(StudyClassifProject,cls).__new__(cls)
        instance.__init__(ID=ID,
                                datas=datas,
                                models=models,
                                metric=metric,
                                project=project)
        # print('cv',instance.cv)
        # instance2=instance
        # if not normal:
        #     instance2=vizHelper(instance)
        return vizHelper(instance) if not normal else instance

