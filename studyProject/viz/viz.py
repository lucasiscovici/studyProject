import cufflinks_study as cf
cf.go_offline(connected=False)

import pandas as pd
from interface import Interface,implements
from ..utils import get_args, isinstanceBase, isinstance
from ..utils import merge,  T,F, dicoAuto, zipl, StudyClass
import numpy as np
import plotly_study.express as pex
class Viz:
    def __init__(self,obj):
        self.obj=obj

    #fnCount (normalize:Boolean)
    def plot_bar_count(self,fnCount,title="Répartition des labels",percent=False,
                                allsize=16,
                                titlesizeplus=2,
                                roundVal=2,
                                maxBars=20,
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
                                fn_kwargs=dict(),
                                plot_kwargs=dict()):

        _addLabels_kwargs=dict(textposition="auto",textfont=dict(color="white",size=allsize))
        if outside:
            _addLabels_kwargs=dict(textposition="outside",textfont=dict(color="black",size=allsize))
        addLabels_kwargs=merge(_addLabels_kwargs,addLabels_kwargs,add=F)

        _fn_kwargs=dict(normalize=percent)
        fn_kwargs=merge(_fn_kwargs,fn_kwargs,add=F)


        nam="x" if xTitle is None else xTitle
        namy="Nombre" if yTitle is None else yTitle

        dio=dicoAuto[["xaxis","yaxis"]][['tickfont','titlefont']].size==allsize
        dio["font"]=dict(size=allsize+titlesizeplus)
        _plot_kwargs=dict()

        data=StudyClass()
        plot_kwargs=merge(_plot_kwargs,plot_kwargs,add=F)
        cb=fnCount(**fn_kwargs)[:maxBars]
        fig=pex.bar(cb.to_frame(),y=cb.name,**plot_kwargs).update_layout(title=title,xaxis_title=nam,
                         yaxis_title=namy,**dio).update_config(toImageButtonOptions=dict(filename=filename))
        setattr(data,"bar_count_percent" if percent else "bar_count",cb)

        if addLabels:
            if addLabelsPercent and not percent:
                fn_kwargs2=merge(fn_kwargs,dict(normalize=True),add=F)
                cb2=fnCount(**fn_kwargs2)[:maxBars]
                fig.data[0].text= list(map(lambda x:"{} ({}%)".format(x[0], np.round((x[1]*100),roundVal)),zipl(cb.values,cb2.values))) 
                setattr(data,"bar_count_percent",cb2)
            elif addLabelsBrut and percent:
                fn_kwargs2=merge(fn_kwargs,dict(normalize=False),add=F)
                cb2=fnCount(**fn_kwargs2)[:maxBars]
                fig.data[0].text= list(map(lambda x:"{} ({}%)".format(x[0], np.round((x[1]*100),roundVal)),zipl(cb2.values,cb.values))) 
                setattr(data,"bar_count",cb2)
            else:
                fig.data[0].text=cb
            for k,v in addLabels_kwargs.items():
                setattr(fig.data[0],k,v)

        fig.update_layout(margin=dict(t=60))
        fig.update_config(editable=T,
                       displayModeBar=T,
                       displaylogo=F,
                       filename=filename)
        if asImg:
            fig=cb.iplot(data=fig,filename=filename,asImage=True)
            return fig
        if returnData:
            if showFig:
                fig.show()
            return StudyClass(data=data,fig=fig)

        return fig if showFig else StudyClass(fig=fig)

import functools

def catch_exception_viz(f,obj,names,kw="me"):
    @functools.wraps(f)
    def func(*args, **kwargs):
        # print(names)
        try:
            if kw in names:
                kwargs[kw]=obj
            return f(*args, **kwargs)
        except Exception as e:
            try:
                return f(*args, **kwargs)
            except Exception as e2:
                print(str(e))
                print(str(e2))
                raise e
    return func

def vizGet(o):
    return o.__curr if o.__class__.__name__ == "vizHelper"  else o
def catch_exception_viz2(f,obj,names,kw="me"):
    @functools.wraps(f)
    def func(*args, **kwargs):
        # print(names)
        try:
            if kw in names:
                kwargs[kw]=obj
            if "_obj" in kwargs and kwargs["_obj"]:
                del kwargs["_obj"]
                return f(*args, **kwargs)
            # if "vh" in args and args["vh"]:
            return vizHelper(obj,f(*args, **kwargs),realNone=True)
            # return f(*args, **kwargs)
        except Exception as e:
            try:
                return f(*args, **kwargs)
            except Exception as e2:
                print(str(e))
                print(str(e2))
                raise e
    return func
# class vizHelperMeta(type):
#     def __instancecheck__(self, other):
#         return isinstance(self.__curr,typ)
import copy
class vizHelper(object):
    def __init__(self, arg, curr=None, realNone=False):
        self.__obj = arg
        self.__curr = (None if realNone else arg) if curr is None else curr 

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        # cls = self.__class__
        # result = cls.__new__(cls)
        # memo[id(self)] = result
        # for k, v in self.__dict__.items():
        #     setattr(result, k, copy.deepcopy(v, memo))
        return self

    def __getstate__(self):
        return dict(obj=self.__obj,
                    curr=self.__curr)

    def __setstate__(self,i):
        self.__obj=i["__obj"]
        self.__curr=i["__curr"]

    def __meme(self,rep):
        # print(callable(rep))
        # print(type(rep).__name__)
        # print(callable(rep) and (type(rep).__name__ in ["function","method"]))
        if callable(rep) and (type(rep).__name__ in ["function","method","builtin_function_or_method"]) :
            ng=get_args(rep).names
            # print(ng)
            if "me" not in ng:
                # print("withNotME")
                return catch_exception_viz2(rep,self.__obj,ng)
            # print("withME")
            return catch_exception_viz(rep,self.__obj,ng)
        return vizHelper(self.__obj,rep)

    def __getattr__(self,k):
        # print("getattr")
        if k=="_vizHelper__obj" or k=="_vizHelper__curr" or k=="_vizHelper__meme":
            k=k[len("_vizHelper"):]
        # print(k)
        if k.startswith('___') and k.endswith('___'):
            return self.__meme(getattr(self.__curr,k))

        if k.startswith('__') and k.endswith('__'):
            return getattr(self,k)

        if k=="__curr" or k=="__obj" or k=="__meme" or k=="_instancecheck" or k=="__getstate__" or k=="__setstate__":
            return getattr(self,k)
        return self.__meme(getattr(self.__curr,k))

    def __setattr__(self,k,v):
        # print("setattr")
        # print(k)
        if k=="_vizHelper__obj" or k=="_vizHelper__curr" or k=="_vizHelper__meme":
            k=k[len("_vizHelper"):]
        if k=="__curr" or k=="__obj" or k=="__meme" or k=="_instancecheck" or k=="__getstate__" or k=="__setstate__":
            object.__setattr__(self,k,v)
        else:
            # print(k)
            setattr(self.__curr,k,v)
        # return self.__meme(getattr(self.__curr,k))
    def __dir__(self):
        return self.__curr.__dir__()
    
    def __repr__(self):
        if self.__curr is None:
            return ""
        try:
            return self.__curr.__repr__()
        except:
            return object.__repr__(self.__curr)

    def _getMe(self):
        return self.__curr
    def __getitem__(self,k):
        if isinstance(self.__curr,dict):
            return self.__meme(self.__curr.get(k))
        return self.__meme(self.__curr[k])

    def __iter__(self):
        return iter(self.__curr)
        
    def _instancecheck(self, typ):
        return isinstanceBase(self.__curr,typ)

    def __len__(self):
        return len(self.__curr)

def disable_plotly_in_cell():
    import IPython
    get_ipython().events.unregister('pre_run_cell', enable_plotly_in_cell)

def enable_plotly_in_cell():
    import IPython
    from plotly_study.offline import init_notebook_mode
    display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
    init_notebook_mode(connected=False)

def plotly_google_colab():
    get_ipython().events.register('pre_run_cell', enable_plotly_in_cell)

import plotly_study.figure_factory as ff
def pdViz(self):
    def to_heatmap(colorscale="Greys",reversescale=False):
        mat=ff.create_annotated_heatmap(z=self.values,x=self.columns.tolist(),y=self.index.tolist(),
                                        annotation_text=self.values,colorscale=colorscale,reversescale=reversescale)
        return heatmap_to_grid(mat)
    return StudyClass(to_heatmap=to_heatmap)
        
#pd.DataFrame.viz = property(lambda self: pdViz(self))
def heatmap_to_grid(hm,paper_bgcol="#F5F6F9",plot_bgcolor="black",linewidth=2,linecolor="black"):
    hm.update_layout(paper_bgcolor= paper_bgcol,
                          plot_bgcolor= plot_bgcolor,
                            xaxis=dict(side="bottom",linewidth= linewidth,mirror=True,
                                showgrid=F,linecolor = linecolor,zeroline=F,showline=T),
                            yaxis=dict(linewidth= linewidth,mirror=True,showgrid=F,linecolor = linecolor,
                                zeroline=F,showline=T))

