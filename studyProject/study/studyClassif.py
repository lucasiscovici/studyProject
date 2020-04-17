from ..base import  BaseSupervise, Metric, DatasSupervise, Models, CrossValidItem, Base
import os
from ..utils import getStaticMethodFromObj, getsourceP, getStaticMethodFromCls, isNumpyArr, listl, zipl, T, F, StudyClass, isPossible, rangel
from abc import ABC, abstractmethod
from interface import implements, Interface
from ..base import DatasSupervise, Datas, factoryCls
import warnings

import warnings as warning
import pandas as pd
import numpy as np
from studyPipe.pipes import * 
from studyPipe import df_
from ..viz.viz import vizHelper
from ..utils import isinstanceBase, isinstance, studyDico,  namesEscape
from typing import Dict
from ..viz import sameErrorsViz

from sklearn.dummy import DummyClassifier

class DatasClassif_ClassBalance:
    def class_balance(self,normalize=True,name="Class Balance"):
        df=self.y.value_counts(normalize=normalize).set_name(name)
        return df

    def table_class_balance(self,normalize=True,name="Class Balance",attr=None,title="Class Balance of {}"):
        n=self.attr if attr is None else attr
        f=self.class_balance(normalize,name).set_name(n).to_frame().T.table_plot().add_title(title.format(n))
        return f

class DatasSuperviseClassif_ClassBalance:
    def class_balance(self,normalize=False):
        rep=(
            self 
            | ((__.dataTrain,__.dataTest) |_funs_| listl )
            | ((__,["dataTrain","dataTest"]) |_funs_| zipl )
            | _ftools_.mapl(__[0].class_balance(normalize=normalize).to_frame(__[1]))
            | (pd.concat |_funsInv_| dict(objs=__,axis=1))
        )
        return rep
    def table_class_balance(self,normalize=True,title="Class Balance of {}"):
        # n=self.attr if attr is None else attr
        n="datas"
        f=self.class_balance(normalize).table_plot().add_title(title.format(n))
        return f

class DatasClassif(Datas,DatasClassif_ClassBalance):
    EXPORTABLE=["cat"]
    EXPORTABLE_ARGS=dict(underscore=False)
    # y must be a series or a dataframe

    def __init__(self,X=None,y=None,cat=None,ID=None, *args,**xargs):
        super().__init__(X,y,*args,**xargs,ID=ID)
        self.cat=cat
        self.init()

    def init(self,*args,**xargs):

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
            cat=namesEscape(self.cat)
            if cat != self.cat:
                self.y=self.y.cat.rename_categories(dict(zip(self.cat,cat)))
        super().init(*args,**xargs)

factoryCls.register_class(DatasClassif)


class DatasSuperviseClassif(DatasSupervise,DatasSuperviseClassif_ClassBalance):
    EXPORTABLE=["dataTrain","dataTest"]
    D=DatasClassif
    def __init__(self,dataTrain:DatasClassif=None,dataTest:DatasClassif=None,ID=None):
        super().__init__(ID=ID)
        self.dataTrain=dataTrain
        self.dataTest=dataTest

factoryCls.register_class(DatasSuperviseClassif)
class StudyClassif_:
    isClassif=True
    def computeCV(self,cv=3,random_state=42,shuffle=True,classifier=True,
                 nameCV=None,recreate=False,parallel=True,metric=Metric("accuracy"),
                 models=None,**xargs):
        return super().computeCV(cv,random_state,shuffle,classifier,nameCV,recreate,parallel,metric,models,**xargs)

from ..base.base import CvResultats, CvSplit
class CvResultatsClassif_ClassificationReport:

    def classification_report(self,y_true="y_train",namesY="train_datas",returnNamesY=False,transpose=True,skip_support=True,orderCol=["precision","recall","f1-score"],me=None):
        obj=self
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=lambda:getattr(me,namesY).cat
                namesY= namesY() if isPossible(namesY) else None
        elif isinstance(y_true,str) or isinstance(namesY,str):
            me=obj.papa.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat

        ff2=classification_report(vizGet(y_true),vizGet(self.preds.Val.sorted),output_dict=True)
        namesY= rangel(len(np.unique(y_true))) if namesY is None else namesY
        ff2=pd.DataFrame(ff2)
        if skip_support:
            ff2=ff2[:-1]
        if transpose:
            ff2=ff2.T
        ff2 = ff2 >> df_.select(*orderCol)
        if returnNamesY:
            return StudyClass(classification_report=ff2,namesY=namesY)
        else:
            return ff2

    def table_classification_report(self,roundVal=3,y_true="y_train",namesY="train_datas",returnNamesY=False,transpose=True,skip_support=True,orderCol=["precision","recall","f1-score"],me=None):
        cf=self.classification_report(y_true=y_true,namesY=namesY,returnNamesY=returnNamesY,transpose=transpose,skip_support=skip_support,orderCol=orderCol,me=me)
        return cf.round(roundVal).table_plot()


class CvResultatsClassif_ConfusionMatrix:
    def confusion_matrix(self,y_true="y_train",namesY="train_datas",
        normalize=True,axis=1,round_=2,relative=False,sum_=True,returnNamesY=False,me=None):
        # print(y_true)
        obj=self
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        elif isinstance(y_true,str) or isinstance(namesY,str):
            me=obj.papa.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat

        ff2=confusion_matrix(y_true,self.preds.Val.sorted)
        ff3=ff2.copy()
        if relative:
            np.fill_diagonal(ff2,0)
        if normalize:
            ff2=np.round(np.divide(ff2,np.sum(ff2,axis=axis,keepdims=True)),round_)*100
        namesY= rangel(len(np.unique(y_true))) if namesY is None else namesY
        if sum_ and relative:
            np.fill_diagonal(ff2,ff3.diagonal())
        p=pd.DataFrame(ff2,columns=namesY).set_axis(namesY,inplace=F)
        if returnNamesY:
            return StudyClass(confusion_matrix=p,namesY=namesY)
        else:
            return p

    def table_confusion_matrix(self,y_true="y_train",namesY="train_datas",normalize=True,axis=1,roundVal=2,returnNamesY=False,me=None):
        s=self.confusion_matrix(y_true=y_true,namesY=namesY,normalize=normalize,axis=axis,round_=roundVal,returnNamesY=returnNamesY,me=me)
        return p.round(roundVal).table_plot()

class CvResultatsClassif(CvResultats,CvResultatsClassif_ClassificationReport,CvResultatsClassif_ConfusionMatrix):

    def getObsConfused(self,classe,predit,lim=10,y_true="y_train",X_true="X_train",namesY="train_datas",me=None):
        obj=self
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(X_true,str):
                X_true=getattr(me,X_true)
            if isinstance(namesY,str):
                namesY=lambda:getattr(me,namesY).cat
                namesY= namesY() if isPossible(namesY) else None
        elif isinstance(y_true,str) or isinstance(X_true,str) or isinstance(namesY,str):
            me=self.papa.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(X_true,str):
                X_true=getattr(me,X_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        X,y,pred = X_true,y_true,self.preds.Val.sorted
        indu=np.where((y==classe) & (predit==pred))
        # print(indu)
        return pd.DataFrame(np.array(X)[(y==classe) & (predit==pred)],index=indu[0]).iloc[:lim,:]
        #return np.array(X)[(y==classe) & (predit==pred)][:lim]

    def getErrors(self):
        err=self.preds.Val.sorted
        errors=err!=self.papa.papa.y_train
        return StudyClass(errors=errors,nb=errors.sum(),perc=errors.mean())
from sklearn.metrics import classification_report, confusion_matrix
from ..viz import vizGet
factoryCls.register_class(CvResultatsClassif)
class CVIClassif_ConfusionMatrix:
    def confusion_matrix(self,y_true="X_train",namesY="train_datas",normalize=True,mods=[],me=None,*args,**xargs):
        # print(y_true)
        obj=self
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        elif isinstance(y_true,str) or isinstance(namesY,str):
            me=obj.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        modsN=list(self.resultats.keys())
        models=self.resultats
        if len(mods)>0:
            mods_ = [i if isStr(i) else modsN[i] for i in mods]
            models= {i:self.resultats[i] for i in mods_}
        r=studyDico({k:v.confusion_matrix(y_true,namesY,normalize,*args,**xargs) for k,v in models.items()}) 
        return r

    from itertools import combinations

    def getCvPreds(self,type_="Val",isSorted=True):
        cv=self
        return {k:getattr(getattr(i.preds,type_),"sorted") if isSorted else getattr(getattr(i.preds,type_),"original") for k,i in cv.resultats.items()}

    @property
    def sameErrors(self):
        from itertools import combinations
        y_test=self.papa.y_train
        if np.ndim(y_test)==2:
            raise NotImplemented("MultiLabelNotImplemented")
            #return [diff_classif_models2([ m[:,i] for m in models],X_test,y_test[:,i],names) for i in range(y_test.ndim)]
        models=self.getCvPreds()
        X_test=self.papa.X_train
        names=self.papa.namesModels
        kkd=[ (y_test!=i).values for k,i in models.items()] 
        # print(kkd)
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
        rep=StudyClass(sameErrors=pd.DataFrame(o,columns=names,index=names),
                          sameErrorsPerc=pd.DataFrame(o2,columns=names,index=names),
                          sameErrorsBrut=o3,
                          errors=kkd)
        rep.viz=property(sameErrorsViz)
        return viz

    def getErrors(self,mods=[]):
        modsN=list(self.resultats.keys())
        models=self.resultats
        if len(mods)>0:
            mods_ = [i if isStr(i) else modsN[i] for i in mods]
            models= {i:self.resultats[i] for i in mods_}

        r={k:v.getErrors() for k,v in models.items()}
        return r
class CVIClassif(CVIClassif_ConfusionMatrix):

    def getObsConfused(self,classe,predit,mods=[],lim=10,y_true="y_train",X_true="X_train",namesY="train_datas",me=None):
        obj=self
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(X_true,str):
                X_true=getattr(me,X_true)
            if isinstance(namesY,str):
                namesY=lambda:getattr(me,namesY).cat
                namesY= namesY() if isPossible(namesY) else None
        elif isinstance(y_true,str) or isinstance(X_true,str) or isinstance(namesY,str):
            me=obj.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(X_true,str):
                X_true=getattr(me,X_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        modsN=list(self.resultats.keys())
        models=self.resultats
        if len(mods)>0:
            mods_ = [i if isStr(i) else modsN[i] for i in mods]
            models= {i:self.resultats[i] for i in mods_}
        r=studyDico({k:v.getObsConfused(classe,predit,lim=lim,y_true=y_true,X_true=X_true,namesY=namesY,me=me) for k,v in models.items()}) 
        return r

class CrossValidItemClassif(CrossValidItem,CVIClassif):
    EXPORTABLE=["resultats"]
    def __init__(self,ID:str=None,cv:CvSplit=None,resultats:Dict[str,CvResultatsClassif]={},
                args:Dict=None):
        super().__init__(
                ID=ID,
                resultats=resultats,
                args=args,
                cv=cv
            )
        # self.resultats=resultats

factoryCls.register_class(CrossValidItemClassif)

class BSClassif:
    def confusion_matrix(self,normalize=True,cvs=None):
        cvs_=[] if cvs is None else (list(cvs.keys()) if isinstance(cvs,dict) else cvs)
        cvKeys=list(self._cv.keys())
        cv_=self._cv
        y_true=self.y_train
        namesY=self.train_datas.cat
        if len(cvs_)>0:
            cv_names = [i if isStr(i) else cvKeys[i] for i in cvs_]
            cv_res= {i:cv_[i] for i in cv_names}
        else:
            cv_res=cv_
        if isinstance(cvs,dict):
            cvD={i:cvs[cvs_[i_]] for i_, i in enumerate(cv_names)}
            r=studyDico({k:v.confusion_matrix(y_true,namesY,normalize,mods=cvD[k]) for k,v in cv_res.items()}) 
        else:
            r=studyDico({k:v.confusion_matrix(y_true,namesY,normalize) for k,v in cv_res.items()}) 

        return r
    def addDummiesModels(self,  dummiesModels=[DummyClassifier(),DummyClassifier(strategy="most_frequent"),DummyClassifier(strategy="uniform")],
                                namesDummiesModels=["DummyStratified","DummyMostFreq","DummyUniform"],
                                nameCV=None,
                                verbose=1,
                                *args,**xargs):
        if self._nameCvCurr is None:
            self.setModels(dummiesModels,
                          names=namesDummiesModels)
            self.computeCV(verbose=verbose,nameCV=nameCV,*args,**xargs)
        else:
            self.addModelsToCurrCV(dummiesModels,names=namesDummiesModels,nameCV=nameCV,*args,**xargs)
class BaseSuperviseClassif(StudyClassif_,BaseSupervise,BSClassif):
    # @abstractmethod
    EXPORTABLE=["datas","cv"]
    EXPORTABLE_ARGS=dict(underscore=True)

    cvrCls=CvResultatsClassif
    cviCls=CrossValidItemClassif
    def __init__(self,ID=None,datas:DatasSuperviseClassif=None,
                    models:Models=None,metric:Metric=None,
                    cv:Dict[str,CrossValidItemClassif]=studyDico({}),nameCvCurr=None,
                    *args,**xargs):
        super().__init__(ID,datas,models,metric,cv,nameCvCurr,*args,**xargs)
        # self._datas=datas
        # self._cv=cv
        # self._isClassif=True

factoryCls.register_class(BaseSuperviseClassif)

class StudyClassif(BaseSuperviseClassif):
    def __init__(self,
                 ID=None,
                 datas:DatasSuperviseClassif=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy"),
                 cv:Dict[str,CrossValidItemClassif]=None,
                 nameCvCurr=None,dejaINIT=False,normal=True):
        # if not dejaINIT:
        if cv is None:
            cv=studyDico({},papa=self,addPapaIf=lambda c:instance(c,Base),attr="_cv")
        super(StudyClassif,self).__init__(
            ID=ID,
            datas=datas,
            models=models,
            metric=metric,
            cv=cv,
            nameCvCurr=nameCvCurr
        )
        self.init()
    # def __new__(cls,
    #              ID=None,
    #              datas:DatasSuperviseClassif=None,
    #              models:Models=None,
    #              metric:Metric=Metric("accuracy"),
    #              cv:Dict[str,CrossValidItem]=None,
    #              nameCvCurr=None,
    #              normal=True):
    #     instance= super(StudyClassif,cls).__new__(cls)
    #     if not normal:
    #         instance.__init__(ID=ID,
    #                                 datas=datas,
    #                                 models=models,
    #                                 metric=metric,
    #                                 cv=cv,
    #                                 nameCvCurr=nameCvCurr)
    #     return instance.vh if not normal else instance
# from ..project.project import BaseSuperviseClassifProject

