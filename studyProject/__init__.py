from .base import *
from .utils import *
from .study import *
from .project import *
from .helpers import *
factoryCls.register_class(StudyProject)
factoryCls.register_class(StudyClassif)
factoryCls.register_class(StudyClassifProject)
factoryCls.register_class(StudyReg)

# print(factoryCls._classes)