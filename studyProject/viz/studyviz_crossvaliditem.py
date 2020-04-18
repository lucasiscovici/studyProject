from . import Viz
from ..utils import F,T
from studyPipe import df,X
import pandas as pd
import numpy as np
import plotly_study.express as px 
class Study_CrossValidItem_Viz(Viz):
    def plot_resultatsSummary(self,with_std=False):
        obj=self.obj
        if not with_std:
            e=(
                obj.resultatsSummary(withStd=F).T.iplot(kind="scatter",mode="markers",asFigure=T)
                .update_config(plotlyServerURL="") 
                .update_layout(width=600)
            )
        else:
            g=(
                    obj.resultatsSummary(withStd=T,withStdCol=T).T.rename_axis(index="model").reset_index()  

                )
            dff=(g.dropCols(["Tr(std)","Val(std)"])>> df.gather("Var","Values",["Tr","Val"])).assign(ci=pd.concat([g.loc[:,"Tr(std)"],g.loc[:,"Val(std)"]],ignore_index=T))

            e=(px.scatter(dff, x="model",y="Values",color="Var",
                error_y="ci"))
        return e
