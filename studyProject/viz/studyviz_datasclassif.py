from . import Viz
from interface import implements
from ..utils import merge,showWarningsTmp,  df, secureAddArgs, removeBadArgs , _get_and_checks,  T,F, dicoAuto, zipl, _get_name, StudyClass, addMethodToObj, mpld3_study, display, display_html, mpld3_utils, mpl_to_plotly, _get_dtype_and_data
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
from collections.abc import Iterable
import types
import plotly_study.express as pex
import matplotlib.pyplot as plt
import seaborn as sns
def createHistMultiBins(self,x,color,colorI,by,byI,nbins,title=None,ncols=3,viz="mpld3"):
        datas=self
        train_df=datas
        catBy=byI.cat.categories
        catColor=colorI.cat.categories
        ncols=min(ncols,len(catBy))
        n_rows = (len(catBy) - 1) // ncols + 1
        fig, axes = plt.subplots(nrows=n_rows, ncols=ncols,figsize=(12, 4), constrained_layout=True);
        if len(nbins)!=len(catBy):
            raise Exception("nbins ERROS")
        #d=[]
        fig.suptitle(title, fontsize=16)
        for i_,(i,bi) in enumerate(zip(catBy,nbins)):
            d=datas[datas[by]==i]
            ax=axes[i_]
            for j in catColor:
                o=d[d[color]==j]
                axo = sns.distplot(getattr(o,x).dropna(), bins=bi, label = j, ax = ax, kde =False);
                #axo.legend(name=title)
                ##handles, labels = axo.get_legend_handles_labels()
                #handles.append(mpatches.Patch(color='none', label=j))
                #axo.legend(handles=handles)
            axo.set_title(i);

        #mpl_to_plotly(fig)
        if viz == "mpld3":
            #plt.text(0.5, 0.99, title, horizontalalignment='center', verticalalignment='top')
            mpld3_utils.connect_mpld3(mpld3_utils.add_hover_label_to_rect(axes)+mpld3_utils.add_select_filter(axes),fig);
            mpld3_utils.addSaveButtonMpld3();
            display_html("""
            <script>
            var __origDefine = define;
            define = null;
            </script>
            <style>
                     .legend{
                     transform: translateX(-150px);
                     }
                    </style>
                    
            <h2>${title}</h2> 
            """.replace("${title}",title))
            
            return addMethodToObj(mpld3_study.display(),"show",lambda self,*args,**xargs: display(self))
        if viz == "mpl":
            return StudyClass(fig=fig,axes=axes,show=lambda *args,**xargs:plt.show())
        if viz == "plotly":
            return mpl_to_plotly(fig)
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
        


    def plot_pointplot(selfo,x,y=None,by=None, color=None,side="row",*args,**xargs):
        self=selfo.obj
        row=_get_name(x)
        datas=self.get()
        yName=self.y.name
        if row not in datas.columns:
            raise Exception(f"'{row}' not in datas")
        
        if y is not None:
            y=_get_name(y)
            if y not in datas.columns:
                raise Exception(f"'{y}' not in datas")
        else:
            y=yName
        
        if by is not None:
            by=_get_name(by)
            if by not in datas.columns:
                raise Exception(f"'{by}' not in datas")
        
        if color is not None:
            color=_get_name(color)
            if color not in datas.columns:
                raise Exception(f"'{color}' not in datas")

        xargs2=dict()
        xargs2[side]=by
        FacetGrid = sns.FacetGrid(datas, size=4.5, aspect=1.6,*args,**xargs2,**xargs)
        FacetGrid.map(sns.pointplot, x, y, color, palette=None,  order=None, hue_order=None )
        FacetGrid.add_legend()
        rep=mpl_to_plotly(d.fig).update_layout(title_text=f"PointPlot with x: {x}, y: {y}, by: {by}, color: {color}", margin=dict(t=75))
        return rep


    def plot(self,x=None,y=None,color=None,by=None,addTargetAuto=False,types=None,update_layout=None,roundVal=2,targetMode=None,*args,**xargs):
        datas=self.obj.get()
        target=self.obj.y.name
        targetI=self.obj.get() >> df.pull(target)
        xIType,yIType,colorIType, byIType = None,None,None,None
        x,y,color,by=_get_and_checks([x,y,color,by],datas,[None,None,None,None,False])
        
        if (all([i is None for i in [x,y,color,by]])): #or all([i is None for i in [x,y]])) and addTargetAuto == False
            raise Exception("plot impossible")
            
        if all([i is None for i in [x,y,color,by]]):
            types= types if types is not None else "bar"
            if types in ["hist","histogram","bar"]:
                argsx=xargs
                if not secureAddArgs(self.obj.viz.plot_class_balance,xargs):
                    with showWarningsTmp:
                        warnings.warn(f"""
                        error in {xargs} not in self.obj.viz.plot_class_balance""")
                    argsx=removeBadArgs(self.obj.viz.plot_class_balance,argsx)
                return self.obj.viz.plot_class_balance(title=f"Counts of '{target}' target variable",yTitle="Count",**argsx)
            raise NotImplementedError(f"types '{types}' not again available")
        
        if targetMode is not None and targetMode  in ["x","y","color","by"]:
            if targetMode=="by":
                by=target
            elif targetMode=="color":
                color=target
            elif targetMode =="x":
                x=target
            elif targetMode=="y":
                y=target
            #exec(f"{targetMode}='{target}'",locals(),globals())
        if x is not None:
            xI,xIType=_get_dtype_and_data(x,datas)
            
        if y is not None:
            yI,yIType=_get_dtype_and_data(y,datas)
            
        if color is not None:
            colorI,colorIType=_get_dtype_and_data(color,datas)
            
        if by is not None:
            byI,byIType=_get_dtype_and_data(by,datas)
        
        if all([i is None for i in [y,color,by]]) and x is not None and addTargetAuto == False:
            if pd.api.types.is_numeric_dtype(xIType):
                types=types if types is not None else "histogram"
                if types in ["hist","histogram"]:
                    argsx=xargs
                    if not secureAddArgs(pex.histogram,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.histogram""")
                        argsx=removeBadArgs(pex.histogram,argsx)
                        
                    return pex.histogram(datas,x=x,**argsx)
                if types in ["bar"]:
                    argsx=xargs
                    if not secureAddArgs(pex.bar,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.bar""")
                        argsx=removeBadArgs(pex.bar,argsx)
                    return pex.bar(datas,x=x,**argsxg)
                if types in ["points","scatter"]:
                    sortOk=xargs.pop("sort",False)
                    x_=np.sort(xI) if sortOk else x
                    sort_="sorted" if sortOk else ""
                    title_=f"{types} plot  of '{x}' {sort_}"
                    argsx=xargs
                    if not secureAddArgs(pex.scatter,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.scatter""")
                        argsx=removeBadArgs(pex.scatter,argsx)
                    return pex.scatter(datas,x=x_,title=title_,**argsx).update_layout(yaxis_title="Range",xaxis_title=x)
                raise NotImplementedError(f"types '{types}' not again available")
            elif pd.api.types.is_categorical_dtype(xIType):
                types=types if types is not None else "bar" 
                if types in ["hist","histogram","bar"]:
                    catOrder=xI.cat.categories
                    #print(catOrder)
                    #print(xI.value_counts())
                    def fnCount(*args,**xargs):
                        rep=xI.value_counts(*args,**xargs).sort_index(level=catOrder)
                        #rep=rep.to_frame().rename_cols(["Count"])
                        return rep
                    #fnCount=lambda *args,**xargs: xI.value_counts(*args,**xargs).sort_index(level=catOrder).as_frame()
                    argsx=xargs
                    if not secureAddArgs(self.plot_bar_count,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in self.plot_bar_count""")
                        argsx=removeBadArgs(self.plot_bar_count,argsx)
                    return self.plot_bar_count(fnCount,lambda *args,**xargs:xI.value_counts().sort_index(level=catOrder).to_frame().rename_cols(["count"]).sort_index(level=catOrder).reset_index().rename_cols([x,"Count"]),ind="x",plot_kwargs=dict(y="Count",barmode="overlay",category_orders=dict(zip([x],[list(catOrder)]))),cbName=x,**argsx).update_layout(yaxis_title="Count",xaxis_title=x,title=f"Count by '{x}'")
            raise NotImplementedError(f"types '{types}' not again available")
        
        if all([i is None for i in [x,color,by]]) and y is not None and addTargetAuto == False:
            if pd.api.types.is_numeric_dtype(yIType):
                types=types if types is not None else "scatter"
                if types in ["points","scatter"]:
                    sortOk=xargs.pop("sort",False)
                    y_=np.sort(yI) if sortOk else yI
                    sort_="sorted" if sortOk else ""
                    title_=f"{types} plot  of '{y}' {sort_}"
                    argsx=xargs
                    if not secureAddArgs(pex.scatter,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.scatter""")
                        argsx=removeBadArgs(pex.scatter,argsx)
                    return pex.scatter(datas,y=y_,title=title_,**argsx).update_layout(xaxis_title="Range",yaxis_title=y)
            raise NotImplementedError(f"when only y and not target and types '{types}' -> not again available")
            
        if all([i is None for i in [x,y,by]]) and color is not None and addTargetAuto == False:
            #print("ici")
            if pd.api.types.is_numeric_dtype(colorIType):
                raise NotImplementedError(f"when only color and types '{colorIType}' -> not again available")
            elif pd.api.types.is_categorical_dtype(colorIType):
                types=types if types is not None else "bar"
                if types in ["bar"]:
                    argsx=xargs
                    if not secureAddArgs(pex.bar,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.bar""")
                        argsx=removeBadArgs(pex.bar,argsx)
                    catOrder=colorI.cat.categories
                    fnCount=lambda *args,**xargs: colorI.value_counts(*args,**xargs).sort_index(level=catOrder)
                    title_=f"Bar plot of '{color}'"
                    rep= pex.bar(datas,x=color,color=color,barmode="overlay",category_orders=dict(zip([color],[list(catOrder)])),**argsx).update_layout(title=title_,xaxis_title=color,yaxis_title="Count")
                    for i,y,z in zip(rep.data,fnCount(),fnCount(normalize=True)*100):
                        i.text=str(y)+" ("+str(np.round(z,roundVal))+"%)"
                        i.hovertemplate=f"{i.name}<br>Count={y}"
                    return rep
                if types in ["count"]:
                    argsx=xargs
                    if not secureAddArgs(self.plot_bar_count,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in self.plot_bar_count""")
                        argsx=removeBadArgs(self.plot_bar_count,argsx)
                    catOrder=colorI.cat.categories
                    fnCount=lambda *args,**xargs: colorI.value_counts(*args,**xargs).sort_index(level=catOrder)
                    return self.plot_bar_count(fnCount,plot_kwargs=dict(color=color),**argsx).update_layout(xaxis_title=color,yaxis_title="Count")
                if types in ["hist","histogram"]:
                    argsx=xargs
                    if not secureAddArgs(pex.histogram,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.histogram""")
                        argsx=removeBadArgs(pex.bar,argsx)
                    
                    return pex.histogram(datas,x=color,color=color,**argsx).update_layout(xaxis_title=color,yaxis_title="Count")
        
        if all([i is None for i in [x,y,color]]) and by is not None and addTargetAuto == False:
            if pd.api.types.is_numeric_dtype(byIType):
                raise NotImplementedError(f"when only color and types '{byIType}' -> not again available")
            elif pd.api.types.is_categorical_dtype(byIType):
                types=types if types is not None else "hist"
                ncols=xargs.pop("ncols",4)
        if y is not None:
             pass
        
        multiple=False
        
    #     if all([i is None for i in [y,by]]) and x is not None and color in [None,target] and addTargetAuto == True:
    #         if pd.api.types.is_numeric_dtype(xIType):
    #             types=types if types is not None else "hist"
    #             if types in ["hist","histogram"]:
    #                 argsx=xargs
    #                 if not secureAddArgs(pex.histogram,xargs):
    #                     with showWarningsTmp:
    #                         warnings.warn(f"""
    #                         error in {xargs} not in pex.histogram""")
    #                     argsx=removeBadArgs(pex.histogram,argsx)
    #                 title_=f"Histogram of '{x}' colored by '{target}'"
    #                 argsx["title"]=title_
    #                 return pex.histogram(datas,x=x,color=target,**argsx)
    #         raise NotImplementedError(f"'{xIType}' -> not again available")
        
        if all([i is None for i in [y,by]]) and x is not None and color is not None and addTargetAuto == False:
            if pd.api.types.is_numeric_dtype(xIType):
                types=types if types is not None else "hist"
                if types in ["hist","histogram"]:
                    argsx=xargs
                    if not secureAddArgs(pex.histogram,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.histogram""")
                        argsx=removeBadArgs(pex.histogram,argsx)
                    if not pd.api.types.is_categorical_dtype(colorIType):
                        raise NotImplementedError(f"color '{colorIType}' and x -> not again available")
                    title_=f"Histogram of '{x}' colored by '{color}'"
                    argsx["title"]=title_
                    catOrder=targetI.cat.categories
                    argsx["category_orders"]=dict(zip([target,color],[list(catOrder),list(colorI.cat.categories)]))
                    return pex.histogram(datas,x=x,color=color,**argsx)
            raise NotImplementedError(f"'{xIType}' -> not again available")
            
        if all([i is None for i in [by,y]]) and x is not None and color is not None and addTargetAuto == True:
            if pd.api.types.is_numeric_dtype(xIType):
                types=types if types is not None else "hist"
                if types in ["hist","histogram"]:
                    argsx=xargs
                    if not pd.api.types.is_categorical_dtype(colorIType):
                        raise NotImplementedError(f"color '{colorIType}' and x -> not again available")
                    nbins=argsx.pop("nbins",None)
                    by=target
                    byI=targetI
                    if nbins is not None:
                        if isinstance(nbins,Iterable) and not isinstance(nbins,str) and len(nbins)>1:
                            if not secureAddArgs(createHistMultiBins,xargs):
                                with showWarningsTmp:
                                    warnings.warn(f"""
                                    error in {xargs} not in createHistMultiBins""")
                                argsx=removeBadArgs(createHistMultiBins,argsx)
                            title_=f"Histogram of '{x}' by '{by}' colored by '{color}'"
                            #print(title_)
                            return createHistMultiBins(datas,x,color,colorI,by,byI,nbins,title=title_,**argsx)
                        else:
                            argsx["nbins"]=nbins
                    if not secureAddArgs(pex.histogram,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.histogram""")
                        argsx=removeBadArgs(pex.histogram,argsx)
                            
                    title_=f"Histogram of '{x}' colored by '{color}' by '{by}'"
                    argsx["title"]=title_
                    catOrder=targetI.cat.categories
                    argsx["category_orders"]=dict(zip([target,color,by],[list(catOrder),list(colorI.cat.categories),list(byI.cat.categories)]))
                    return pex.histogram(datas,x=x,color=color,facet_col=by,**argsx)
            raise NotImplementedError(f"'{xIType}' -> not again available")
            
        if all([i is None for i in [color,y]]) and x is not None and by is not None and addTargetAuto == True:
            if pd.api.types.is_numeric_dtype(xIType):
                types=types if types is not None else "hist"
                if types in ["hist","histogram"]:
                    argsx=xargs
                    nbins=argsx.pop("nbins",None)
                    if nbins is not None:
                        if isinstance(nbins,Iterable) and not isinstance(nbins,str) and len(nbins)>1:
                            if not secureAddArgs(createHistMultiBins,xargs):
                                with showWarningsTmp:
                                    warnings.warn(f"""
                                    error in {xargs} not in createHistMultiBins""")
                                argsx=removeBadArgs(createHistMultiBins,argsx)
                            color=target
                            colorI=targetI
                            title_=f"Histogram of '{x}' by '{by}' colored by '{color}'"
                            #print(title_)
                            return createHistMultiBins(datas,x,color,colorI,by,byI,nbins,title=title_,**argsx)
                        else:
                            argsx["nbins"]=nbins
                    if not secureAddArgs(pex.histogram,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.histogram""")
                        argsx=removeBadArgs(pex.histogram,argsx)
                    title_=f"Histogram of '{x}' by '{by}' colored by '{target}'"
                    argsx["title"]=title_
                    catOrder=targetI.cat.categories
                    argsx["category_orders"]=dict(zip([target,by],[list(catOrder),list(byI.cat.categories)]))
                    return pex.histogram(datas,x=x,color=target,facet_col=by,**argsx)
            raise NotImplementedError(f"'{xIType}' -> not again available")
            

        if all([i is None for i in [x,y,by]]) and color is not None and addTargetAuto == True:
            # print(pd.api.types.is_categorical_dtype(colorIType))
            if pd.api.types.is_numeric_dtype(colorIType):
                raise NotImplementedError(f"when only color and types '{colorIType}' -> not again available")
            elif pd.api.types.is_categorical_dtype(colorIType):
                types=types if types is not None else "bar"
                if types in ["bar"]:
                    argsx=xargs
                    if not secureAddArgs(pex.bar,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.bar""")
                        argsx=removeBadArgs(pex.bar,argsx)
                    catOrder=targetI.cat.categories
                    fnCount=lambda *args,**xargs: targetI.value_counts(*args,**xargs).sort_index(level=catOrder)
                    title_=f"Bar plot of '{target}'"
                    datag=pd.crosstab(targetI,colorI).sort_index(level=catOrder)
                    #print(datag)
                    #rep=datag.iplot(kind="bar")
                    #print(catOrder)
                    #print(dict(zip([target,color],[list(catOrder),list(colorI.cat.categories)])))
                    rep= pex.histogram(datas,x=target,color=color,barmode="group",
                                       category_orders=dict(zip([target,color],[list(catOrder),list(colorI.cat.categories)])),**argsx).update_traces(showlegend=T).update_layout(showlegend=T,title=title_,xaxis_title=target,yaxis_title="Count")
                    #                 for i,y,z in zip(rep.data,fnCount(),fnCount(normalize=True)*100):
    #                     i.text=str(y)+" ("+str(np.round(z,roundVal))+"%)"
    #                     i.hovertemplate=f"{i.name}<br>Count={y}"
                    for j in rep.data:
                        j.legendgroup
                    return rep
                if types in ["count"]:
                    argsx=xargs
                    if not secureAddArgs(self.plot_bar_count,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in self.plot_bar_count""")
                        argsx=removeBadArgs(self.plot_bar_count,argsx)
                    catOrder=colorI.cat.categories
                    fnCount=lambda *args,**xargs: colorI.value_counts(*args,**xargs).sort_index(level=catOrder)
                    return self.plot_bar_count(fnCount,plot_kwargs=dict(color=color),**argsx).update_layout(xaxis_title=color,yaxis_title="Count")
                if types in ["hist","histogram"]:
                    argsx=xargs
                    if not secureAddArgs(pex.histogram,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in pex.histogram""")
                        argsx=removeBadArgs(pex.bar,argsx)
                    return pex.histogram(datas,x=color,color=color,**argsx).update_layout(xaxis_title=color,yaxis_title="Count")
            

        if all([i is None for i in [y]]) and all([i is not None for i in [color,x,by]])  and addTargetAuto == True:
            if pd.api.types.is_categorical_dtype(colorIType) and pd.api.types.is_categorical_dtype(xIType) and pd.api.types.is_categorical_dtype(byIType) :
                types=types if types is not None else "point"
                if types in ["point","pointplot"]:
                    argsx=xargs
                    if not secureAddArgs(self.plot_pointplot,xargs):
                        with showWarningsTmp:
                            warnings.warn(f"""
                            error in {xargs} not in self.plot_pointplot""")
                        argsx=removeBadArgs(self.plot_pointplot,argsx)
                    return self.plot_pointplot(x=x,y=target,color=color,by=by, **argsx)
                raise NotImplementedError(f"when only color and types '{colorIType}' -> not again available")
        raise NotImplementedError(f"when color:'{colorIType}' x:'{xIType}' y: '{yIType}' by: '{byIType}' addTargetAuto: '{addTargetAuto}'  -> not again available")
        # if pd.api.types.is_numeric_dtype(varIType):
        #     types="histogram" if types is None else types
        #     nbins = xargs.pop("bins",None)
        
    def plotWithTarget(self,x=None,y=None,color=None,by=None,types=None,targetMode=None,*args,**xargs):
        return self.plot(x,y,color,by,True,types,targetMode=targetMode,*args,**xargs)

from functools import wraps
def embedPrep(func):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      return func(self.obj.prep,*args,**kwargs)
  return with_logging

from dora_study import Dora
for i in ["plot_feature","explore"]:
    setattr(Study_DatasClassif_Viz,i,embedPrep(getattr(Dora,i)))
