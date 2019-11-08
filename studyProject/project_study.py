from .base import factoryCls, Models, Metric, CvSplit
from .study.studyClassif import CvResultatsClassif, DatasSuperviseClassif, CrossValidItemClassif, StudyClassif_, CVIClassif
from .project.project import CrossValidItemProject, StudyProject,BaseSuperviseProject
from typing import Dict
from .viz.viz import vizHelper

class CrossValidItemClassifProject(CrossValidItemProject,CVIClassif):
    EXPORTABLE=["resultats"]
    def __init__(self,ID:str=None,cv:CvSplit=None,resultats:Dict[str,CvResultatsClassif]={},args:Dict=None,*args_,**xargs):
        super().__init__(ID=ID,cv=cv,resultats=resultats,args=args,*args_,**xargs)
        self.resultats=resultats
factoryCls.register_class(CrossValidItemClassifProject)

class BaseSuperviseClassifProject(BaseSuperviseProject):
    EXPORTABLE=["datas","cv"]
    EXPORTABLE_ARGS=dict(underscore=True)

    def __init__(self,ID=None,datas:DatasSuperviseClassif=None,
                        models:Models=None,metric:Metric=None,
                        cv:Dict[str,CrossValidItemClassifProject]={},
                        project:StudyProject=None,*args,**xargs):
        super().__init__(ID,datas,models,metric,cv,project,*args,**xargs)
        self._datas=datas
        self._cv=cv
factoryCls.register_class(BaseSuperviseClassifProject)


class StudyClassifProject(StudyClassif_,BaseSuperviseClassifProject):
    def __init__(self,
                 ID=None,
                 datas:DatasSuperviseClassif=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy"),
                 cv:Dict[str,CrossValidItemClassifProject]=None,
                 nameCvCurr=None,
                 dejaINIT=False,
                 project=None): 
        if not dejaINIT:
            if cv is None:
                cv={}
            super().__init__(
                ID=ID,
                datas=datas,
                models=models,
                metric=metric,
                cv=cv,
                nameCvCurr=nameCvCurr,
                project=project
            )
            self.init()
    def __new__(cls,
                 ID=None,
                 datas:DatasSuperviseClassif=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy"),
                 cv:Dict[str,CrossValidItemClassif]=None,
                 nameCvCurr=None,
                 project=None,normal=True):
        instance= super(StudyClassifProject,cls).__new__(cls)
        if not normal:
            instance.__init__(ID=ID,
                                    datas=datas,
                                    models=models,
                                    metric=metric,
                                    cv=cv,
                                    nameCvCurr=nameCvCurr,
                                    project=project)
        return vizHelper(instance) if not normal else instance