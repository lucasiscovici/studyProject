from .base import *
from .utils import *
from .study import *
from .project import *
from .helpers import *
from .viz import *
factoryCls.register_class(StudyProject)
factoryCls.register_class(StudyClassif)
# factoryCls.register_class(StudyReg)

factoryCls.register_class(Study_CvResultatsClassif_Viz)
factoryCls.register_class(Study_DatasClassif_Viz)
factoryCls.register_class(Study_DatasSuperviseClassif_Viz)
factoryCls.register_class(Study_CVIClassif_Viz)
factoryCls.register_class(Study_BaseSuperviseClassif_Viz)
# factoryCls.register_class(Study_BaseSuperviseProject_Viz)
factoryCls.register_class(Study_StudyClassif__Viz)
factoryCls.register_class(Study_CrossValidItem_Viz)
# factoryCls.register_class(Study_StudyClassifProject_Viz)
# print(factoryCls._classes)
 # BaseSuperviseProject

from .project_study import *

factoryCls.register_class(StudyClassifProject)