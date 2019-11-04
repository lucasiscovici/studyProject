from .base import *
from .utils import *
from .project import *
from .study import *
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