from ..base import factoryCls
from .viz import Viz
from .pandasNewMethodsPlots import *

from .studyviz_datas import Study_Datas_Viz
from .studyviz_datasupervise import Study_DatasSupervise_Viz
from .studyviz_cvresultats import Study_CvResultats_Viz
from .studyviz_crossvaliditem import Study_CrossValidItem_Viz
from .studyviz_basesupervise import Study_BaseSupervise_Viz
factoryCls.register_class(Study_CvResultats_Viz)
factoryCls.register_class(Study_Datas_Viz)
factoryCls.register_class(Study_DatasSupervise_Viz)
factoryCls.register_class(Study_CrossValidItem_Viz)
factoryCls.register_class(Study_BaseSupervise_Viz)