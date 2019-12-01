from . import Viz
from interface import implements
from ..utils import merge,  T,F, dicoAuto, zipl, StudyClass
import pandas as pd
import numpy as np


# from ..utils import T,F
class Study_DatasSuperviseClassif_Viz(Viz):
    # @staticmethod
    def plot_class_balance(selfo,title="Répartition des labels Train et Test",percent=False,
                            allsize=16,
                            titlesizeplus=2,
                            addLabels=True,
                            addLabelsPercent=True,
                            addLabelsBrut=True,
                            returnData=False,
                            asImg=False,
                            showFig=True,
                            outside=False,
                            filename="class_balance",
                            xTitle=None,
                            yTitle=None,
                            addLabels_kwargs=dict(),
                            class_balance_kwargs=dict(),
                            plot_kwargs=dict()):
        
        self=selfo.obj
        nam=self.y.name if xTitle is None else xTitle
        namy="Nombre" if yTitle is None else yTitle
        _addLabels_kwargs=dict(textposition="auto",textfont=dict(color="white",size=allsize))
        if outside:
            _addLabels_kwargs=dict(textposition="outside",textfont=dict(color="black",size=allsize))
        addLabels_kwargs=merge(_addLabels_kwargs,addLabels_kwargs,add=F)
    
        _class_balance_kwargs=dict(normalize=percent)
        class_balance_kwargs=merge(_class_balance_kwargs,class_balance_kwargs,add=F)


        dio=dicoAuto[["xaxis","yaxis"]][['tickfont','titlefont']].size==allsize
        dio["font"]=dict(size=allsize+titlesizeplus)
        _plot_kwargs=dict(layout_update=dio,
                                                                      title=title,xTitle=nam,
                         yTitle=namy)
        data=StudyClass()
        plot_kwargs=merge(_plot_kwargs,plot_kwargs,add=F)
        cb=self.class_balance(**class_balance_kwargs)
        fig=cb.iplot(kind="bar",asFigure=True,filename=filename,**plot_kwargs)
        setattr(data,"class_balance_percent" if percent else "class_balance",cb)
        if addLabels:
            if addLabelsPercent and not percent:
                class_balance_kwargs2=merge(class_balance_kwargs,dict(normalize=True),add=F)
                cb2=self.class_balance(**class_balance_kwargs2)
                #print(zipl(cb.values,cb2.values) | _ftools_.mapl(  [__[0],__[1]] %_fun_% list))
                fig.data[0].text= list(map(lambda x:"{} ({}%)".format(x[0], np.round((x[1]*100),2)),zipl(cb.values[:,0],cb2.values[:,0]))) 
                fig.data[1].text= list(map(lambda x:"{} ({}%)".format(x[0], np.round((x[1]*100),2)),zipl(cb.values[:,1],cb2.values[:,1]))) 

                setattr(data,"class_balance_percent",cb2)
            elif addLabelsBrut and percent:
                class_balance_kwargs2=merge(class_balance_kwargs,dict(normalize=False),add=F)
                cb2=self.class_balance(**class_balance_kwargs2)
                # print(cb)
                #print(zipl(cb.values,cb2.values) | _ftools_.mapl(  [__[0],__[1]] %_fun_% list))
                fig.data[0].text= list(map(lambda x:"{} ({}%)".format(x[0], np.round((x[1]*100),2)),zipl(cb2.values[:,0],cb.values[:,0]))) 
                fig.data[1].text= list(map(lambda x:"{} ({}%)".format(x[0], np.round((x[1]*100),2)),zipl(cb2.values[:,1],cb.values[:,1]))) 

                setattr(data,"class_balance",cb2)
            else:
                fig.data[0].text=cb
            for k,v in addLabels_kwargs.items():
                setattr(fig.data[0],k,v)
                setattr(fig.data[1],k,v)
        fig.update_layout(margin=dict(t=60))
        if asImg:
            fig=cb.iplot(data=fig,filename=filename,asImage=True)
            return fig
        if returnData:
            if showFig:
                fig.show()
            return StudyClass(data=data,fig=fig)
        return fig if showFig else StudyClass(fig=fig)

from functools import wraps
def embedPrep(func):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      return func(self.obj.prep,*args,**kwargs)
  return with_logging

def addMethodsFromSpeedMLPlot():
    from speedml_study import Plot
    fd=Plot.__dict__
    n=[i for i in list(fd.keys()) if not i.startswith("_")] 
    for i in n:
        # print(i)
        setattr(Study_DatasSuperviseClassif_Viz,"plot_"+i,embedPrep(fd[i]))

addMethodsFromSpeedMLPlot()

# print("ici")