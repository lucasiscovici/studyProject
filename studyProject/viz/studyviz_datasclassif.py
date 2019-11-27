from . import Viz
from interface import implements
from ..utils import merge,  T,F, dicoAuto, zipl, StudyClass
import pandas as pd
import numpy as np


class Study_DatasClassif_Viz(Viz):
    # @staticmethod
    def plot_class_balance(self,title="Répartition des labels",percent=False,
                            allsize=16,
                            titlesizeplus=2,
                            addLabels=True,
                            addLabelsPercent=True,
                            addLabelsBrut=True,
                            returnData=False,
                            asImg=False,
                            showFig=True,
                            outside=True,
                            xTitle=None,
                            yTitle=None,
                            filename="class_balance",
                            addLabels_kwargs=dict(),
                            class_balance_kwargs=dict(),
                            plot_kwargs=dict(),*args,**xargs):
        vars_=locals().copy()
        del vars_["self"]
        vars_["fn_kwargs"]=vars_["class_balance_kwargs"]
        del vars_["class_balance_kwargs"]
        vars_["fnCount"]=self.obj.class_balance

        yn=self.obj.y.name
        vars_["xTitle"]= yn if xTitle is None else xTitle

        del vars_["args"]
        del vars_["xargs"]
        return self.plot_bar_count(*args,**vars_,**xargs)
        

from functools import wraps
def embedPrep(func):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      return func(self.obj.prep,*args,**kwargs)
  return with_logging

from dora_study import Dora
for i in ["plot_feature","explore"]:
    setattr(Study_DatasClassif_Viz,i,embedPrep(getattr(Dora,i)))
