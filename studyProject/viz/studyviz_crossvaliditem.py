from . import Viz
from ..utils import F,T
class Study_CrossValidItem_Viz(Viz):
    def plot_resultatsSummary(self):
        obj=self.obj
        e=(
            obj.resultatsSummary(withStd=F).T.iplot(kind="scatter",mode="markers",asFigure=T)
            .update_config(plotlyServerURL="") 
            .update_layout(width=600)
        )
        return e
