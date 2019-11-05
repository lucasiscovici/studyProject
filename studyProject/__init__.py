from .base import *
from .utils import *
from .study import *
from .project import *
from .helpers import *
from .viz import *
factoryCls.register_class(StudyProject)
factoryCls.register_class(StudyClassif)
factoryCls.register_class(StudyClassifProject)
factoryCls.register_class(StudyReg)

factoryCls.register_class(Study_CvResultats_Viz)
factoryCls.register_class(Study_Datas_Viz)
factoryCls.register_class(Study_DatasSupervise_Viz)
factoryCls.register_class(Study_CrossValidItem_Viz)
factoryCls.register_class(Study_BaseSupervise_Viz)
# factoryCls.register_class(Study_BaseSuperviseProject_Viz)
factoryCls.register_class(Study_StudyClassif__Viz)
# factoryCls.register_class(Study_StudyClassif_Viz)
# factoryCls.register_class(Study_StudyClassifProject_Viz)
# print(factoryCls._classes)
 # BaseSuperviseProject

class StudyClassifProject(StudyClassif_,BaseSuperviseClassifProject):
    def __init__(self,
                 ID=None,
                 datas:DatasSuperviseClassif=None,
                 models:Models=None,
                 metric:Metric=Metric("accuracy"),
                 cv:Dict[str,CrossValidItemClassif]=None,
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