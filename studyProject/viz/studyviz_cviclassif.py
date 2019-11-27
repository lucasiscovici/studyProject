import plotly_study.figure_factory as ff
import numpy as np
from ..utils import StudyClass,namesEscape
from . import Viz
from .studyviz_crossvaliditem import Study_CrossValidItem_Viz
from plotly_study.subplots import make_subplots
import cufflinks_study as cf
from cufflinks_study.tools import get_len
from plotly_study.offline import iplot
import plotly_study.graph_objs as go
from functools import reduce
import operator
from ..utils import isStr, T, F, merge, IMG_GRID
from operator import itemgetter
import matplotlib.pyplot as plt

class Study_CVIClassif_Viz(Study_CrossValidItem_Viz):
    def plot_confusion_matrix(self,y_true="y_train",namesY="train_datas",mods=[],normalize=True,addDiagonale=True,colorscale="Greys",
        showscale=True,reversescale=True,size=18,width=500,line_color="red",line_dash="longdash",line_width=6,
        nbCols=3,colFixed=None,shared_xaxes=True,relative=F,axis=1,
                                shared_yaxes=False,vertical_spacing=0.02,horizontal_spacing=0.15,title=None,plots_kwargs={},
                                modelsNames=None,cvName=None,prefixTitle="Confusion Matrix of ",me=None,**plotConfMat_kwargs):
        # print(y_true)
        obj=self.obj
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        elif isinstance(y_true,str) or isinstance(namesY,str):
            me=obj.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat

        modsN=list(obj.resultats.keys())
        models=obj.resultats
        if modelsNames is not None:
            models=dict(zip(modelsNames,models.values()))
        if len(mods)>0:
            mods_ = [i if isStr(i) else modsN[i] for i in mods]
            models= [obj.resultats[i] for i in mods_]
            modelsNames_=[i for i in mods_]
            models=dict(zip(modelsNames_,models)) if modelsNames is None else dict(zip(modelsNames,models))

        namesY= namesEscape(namesY) if namesY is not None else namesY
        if len(models) > 1:
            confMatM=[v.viz.plot_confusion_matrix(y_true,relative=relative,
                namesY=namesY,normalize=normalize,addDiagonale=addDiagonale,onlyConfMat=True,axis=axis,**plotConfMat_kwargs) for v in models.values()]
            # print(confMatM)
            zmin=min(map(itemgetter(0),confMatM))
            zmax=max(map(itemgetter(1),confMatM))
            zmid=None
            plotConfMat_kwargs["zmin"]=zmin
            plotConfMat_kwargs["zmax"]=zmax
            plotConfMat_kwargs["zmid"]=zmid
            plotConfMat_kwargs["dontRescale"]=True
            # print(zmin,zmax,zmid)
        confMatCls={k:v.viz.plot_confusion_matrix(y_true,relative=relative,
                namesY=namesY,normalize=normalize,addDiagonale=addDiagonale,colorscale=colorscale,
                showscale=showscale,reversescale=reversescale,size=size,width=width,
                line_color=line_color,line_dash=line_dash,axis=axis,line_width=line_width,plots_kwargs=plots_kwargs,title=(prefixTitle+"{}").format(k),name="Diag {}".format(i_+1),**plotConfMat_kwargs) 
        for i_,(k,v) in enumerate(models.items())}
        nbCols=min(len(confMatCls),nbCols)
        images_per_row=nbCols
        # images_per_row = min(len(confMatCls), images_per_row)
        n_rows = (len(confMatCls) - 1) // images_per_row + 1

        rowsCol=(len(confMatCls) if colFixed is not None else n_rows,colFixed if colFixed is not None else nbCols)
        titles=[]
        tiplesS=[i.layout.title.text  for i in confMatCls.values()]
        tiplesSAxis=[(i.layout.xaxis.title.text,i.layout.yaxis.title.text)  for i in confMatCls.values()]
        for i in confMatCls.values():
            i.update_layout(xaxis_title="",yaxis_title="")
        subpl=cf.subplots(list(confMatCls.values()),shape=rowsCol,shared_xaxes=shared_xaxes,shared_yaxes=shared_yaxes,
                           horizontal_spacing=horizontal_spacing,
                           vertical_spacing=vertical_spacing,subplot_titles=tiplesS,x_title=tiplesSAxis[0][0],
                           y_title=tiplesSAxis[0][1])
        
        annot=[i.layout.annotations for i in confMatCls.values()]
    
        size=list(confMatCls.values())[0]["layout"]["font"]["size"]

        X=[("x{}".format(j+1),"y{}".format(i+1)) for i in range(rowsCol[0]) for j in range(rowsCol[1])][:len(confMatCls)]
        X=[("x{}".format(i+1),"y{}".format(i+1)) for i in range(len(confMatCls))]
        def modifAnnotRef(annot,xn,yn):
            k=[]
            for i in annot:
                i.xref=xn
                i.yref=yn
                k.append(i)
            return k
        # print(X)
        annot2=reduce(operator.add,[modifAnnotRef(annot[i],x,y) for i,(x,y) in enumerate(X)])
        subpl["layout"]["annotations"]=subpl["layout"]["annotations"]+annot2
        subpl["layout"]["font"]=dict(size=size)

        # subpl["layout"]

        fig= go.Figure(subpl)
        if title is None:
            fig.update_layout(title_text="Confusion Matrix : cv '{}' ({})".format(obj.ID if cvName is None else cvName,"col" if axis==0 else "row"))
        else:
            fig.update_layout(title_text=title)

        # for axis, n in list(get_len(fig).items()):
        #   for u in range(1,n+1):
        #       _='' if u==1 else u
        #       o=0 if axis == "x" else 1
        #       if shared_xaxes and axis=="x" and u > 1:
        #           continue
        #       fig['layout']['{0}axis{1}'.format(axis,_)]["title"]=dict(text=tiplesSAxis[u-1][o])
        
        fig.update_layout(legend=dict(x=0, y=-0.1),
                          legend_orientation="h")

        # print(list(confMatCls.values())[0]["data"])
        for i in fig.data:
            if i.__class__.__name__=="Heatmap":
                i.update(hovertemplate = "<b>%{text}%</b><br>" +
                tiplesSAxis[0][1]+" : %{y}<br>" +
                tiplesSAxis[0][0]+" : %{x}<br>" + "<extra></extra>")
        # fig.update_layout(xaxis_title=tiplesSAxis[0][0],yaxis_title=tiplesSAxis[0][1])

        # fig.update_xaxis(title_text=tiplesSAxis[0])
        # fig.update_yaxis(title_text=tiplesSAxis[1])

        return fig


    def plot_classification_report(self,y_true="y_train",namesY="train_datas",mods=[],normalize=True,addDiagonale=True,colorscale="Greys",
        showscale=True,reversescale=False,size=18,width=500,line_color="red",line_dash="longdash",line_width=6,
        nbCols=3,colFixed=None,shared_xaxes=True,updateEachPlot=dict(),
                                shared_yaxes=False,vertical_spacing=0.02,horizontal_spacing=0.15,title=None,plots_kwargs={},
                                modelsNames=None,cvName=None,prefixTitle="Confusion Matrix of ",me=None,**plotConfMat_kwargs):
        # print(y_true)
        obj=self.obj
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        elif isinstance(y_true,str) or isinstance(namesY,str):
            me=obj.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        modsN=list(obj.resultats.keys())
        models=obj.resultats
        if modelsNames is not None:
            models=dict(zip(modelsNames,models.values()))
        if len(mods)>0:
            mods_ = [i if isStr(i) else modsN[i] for i in mods]
            models= [obj.resultats[i] for i in mods_]
            modelsNames_=[i for i in mods_]
            models=dict(zip(modelsNames_,models)) if modelsNames is None else dict(zip(modelsNames,models))

        namesY= namesEscape(namesY) if namesY is not None else namesY
        if len(models) > 1:
            confMatM=[v.viz.plot_classification_report(y_true,
                namesY=namesY,normalize=normalize,addDiagonale=addDiagonale,onlyConfMat=True) for v in models.values()]
            # print(confMatM)
            zmin=min(map(itemgetter(0),confMatM))
            zmax=max(map(itemgetter(1),confMatM))
            zmid=None
            plotConfMat_kwargs["zmin"]=zmin
            plotConfMat_kwargs["zmax"]=zmax
            plotConfMat_kwargs["zmid"]=zmid
            plotConfMat_kwargs["dontRescale"]=True
            # print(zmin,zmax,zmid)
        confMatCls={k:v.viz.plot_classification_report(y_true,
                namesY=namesY,colorscale=colorscale,
                showscale=showscale,reversescale=reversescale,size=size,width=width,
                line_color=line_color,line_dash=line_dash,line_width=line_width,plots_kwargs=plots_kwargs,title=(prefixTitle+"{}").format(k),name="Diag {}".format(i_+1),**plotConfMat_kwargs).update_layout(updateEachPlot) 
        for i_,(k,v) in enumerate(models.items())}
        nbCols=min(len(confMatCls),nbCols)
        images_per_row=nbCols
        images_per_row = min(len(confMatCls), images_per_row)
        n_rows = (len(confMatCls) - 1) // images_per_row + 1

        rowsCol=(len(confMatCls) if colFixed is not None else n_rows,colFixed if colFixed is not None else nbCols)
        titles=[]
        tiplesS=[i.layout.title.text  for i in confMatCls.values()]
        tiplesSAxis=[(i.layout.xaxis.title.text,i.layout.yaxis.title.text)  for i in confMatCls.values()]
        for i in confMatCls.values():
            i.update_layout(xaxis_title="",yaxis_title="")
        subpl=cf.subplots(list(confMatCls.values()),shape=rowsCol,shared_xaxes=shared_xaxes,shared_yaxes=shared_yaxes,
                           horizontal_spacing=horizontal_spacing,
                           vertical_spacing=vertical_spacing,subplot_titles=tiplesS,x_title=tiplesSAxis[0][0],
                           y_title=tiplesSAxis[0][1])
        
        annot=[i.layout.annotations for i in confMatCls.values()]
    
        size=list(confMatCls.values())[0]["layout"]["font"]["size"]

        X=[("x{}".format(j+1),"y{}".format(i+1)) for i in range(rowsCol[0]) for j in range(rowsCol[1])][:len(confMatCls)]
        X=[("x{}".format(i+1),"y{}".format(i+1)) for i in range(len(confMatCls))]
        def modifAnnotRef(annot,xn,yn):
            k=[]
            for i in annot:
                i.xref=xn
                i.yref=yn
                k.append(i)
            return k
        # print(X)
        annot2=reduce(operator.add,[modifAnnotRef(annot[i],x,y) for i,(x,y) in enumerate(X)])
        subpl["layout"]["annotations"]=(subpl["layout"]["annotations"] if "annotations" in subpl["layout"] else [])+annot2
        subpl["layout"]["font"]=dict(size=size)

        # subpl["layout"]

        fig= go.Figure(subpl)
        if title is None:
            fig.update_layout(title_text="Classification report : cv '{}'".format(obj.ID if cvName is None else cvName))
        
        for axis, n in list(get_len(fig).items()):
            for u in range(1,n+1):
                _='' if u==1 else u
                o=0 if axis == "x" else 1
                # if shared_xaxes and axis=="x" and u > 1:
                    # continue
                fig['layout']['{0}axis{1}'.format(axis,_)]["title"]=dict(text=tiplesSAxis[u-1][o])
                u=fig['layout']['{0}axis{1}'.format(axis,_)].to_plotly_json()
                # print(u)
                fig['layout']['{0}axis{1}'.format(axis,_)]=merge(u,
                    dict(
        linewidth=2,
        linecolor="black",
        mirror=True,
                                showgrid=F,zeroline=F),add=F)

        
        fig.update_layout(legend=dict(x=0, y=-0.1),
                          legend_orientation="h",paper_bgcolor="#F5F6F9",plot_bgcolor="black")


# paper_bgcolor
# plot_bgcolor
        # print(list(confMatCls.values())[0]["data"])
        # for i in fig.data:
        #   if i.__class__.__name__=="Heatmap":
        #       i.update(hovertemplate = "<b>%{text}%</b><br>" +
        #       tiplesSAxis[0][1]+" : %{y}<br>" +
        #       tiplesSAxis[0][0]+" : %{x}<br>" + "<extra></extra>")
        fig.update_layout(xaxis_title=tiplesSAxis[0][0],yaxis_title=tiplesSAxis[0][1])

        # fig.update_xaxis(title_text=tiplesSAxis[0])
        # fig.update_yaxis(title_text=tiplesSAxis[1])

        return fig


    def plot_ObsConfused(self,classes,preds,globalLim=10,globalNbCols=2,
        lim=10,limByPlots=100,elemsByRows=10,nbCols=2,mods=[],title=None,modelsNames=None,filename=None,titleFontsize=19,**plotConfMat_kwargs):
        # from ..helpers import plotDigits
        obj=self.obj

        modsN=obj.papa._models.namesModels
        models=obj.resultats

        if len(mods)>0:
            mods_ = [i if isStr(i) else modsN[i] for i in mods]
            models= [obj.resultats[i] for i in mods_]
            modelsNames_=[i for i in mods_]
            models=dict(zip(modelsNames_,models)) if modelsNames is None else dict(zip(modelsNames,models))

        # namesY= namesEscape(namesY) if namesY is not None else namesY
        confMatM=[v.viz.plot_ObsConfused(classes,preds,lim=lim,limByPlots=limByPlots,
                                        elemsByRows=elemsByRows,returnOK=True,nbCols=nbCols,show=False,**plotConfMat_kwargs) for v in models.values()]

        title = "ObsConfused CV {}".format(obj.ID) if title is None else title
        filename="obj_confused_cv_{}.png".format(obj.ID) if filename is None else filename
        # print(confMatM)
        img=IMG_GRID.grid(confMatM,nbCols=globalNbCols,toImg=True,title=title,titleFontsize=titleFontsize)
        # img.show(figsize=img.figsize,show=True);
        # print(img.data)
        fig=img.show(returnFig=True,show=False,figsize=img.figsize)
        from IPython.display import display_html, HTML
        import mpld3
        # mpld3.enable_notebook()
        display_html(HTML("""
        <style>
        g.mpld3-xaxis, g.mpld3-yaxis {
        display: none;
        }
        </style>
        """))
        # # print(img)
        # # print(img.filename)
        # # print(img.data)
        display_html(HTML("""
            <span style='width:20px;height:20px;position: absolute;' title="Save image as png">
        <a href="data:image/png;base64,{{imgData}}" download="{{filename}}"><img width="20px" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAABmJLR0QA/wD/AP+gvaeTAAAAs0lEQVRIie2WPQ6DMAxGXzJwqIrerN3pORi7cqWwtjegQymyUlMZCIlU8UleIvt7cv4BKuAG9MCQIJ7ABfBEapTkU5xkVC087mMTk4ICskqrkWOdhGntpwJ9OvNuxtgtAMU1mt81F+iRC/S9BfdScVBtrHciAM6/Epds59UqPnW7KMUdp0nee0O8RtbzY9Xk/X9rdIAOUBlQn4ETPNCKAevzYJF8Mlp4f4ca9G/X1gijd/UCDStihJWAousAAAAASUVORK5CYII="></a></span>
            """.replace("{{imgData}}",str(img.data)[2:-1]).replace("{{filename}}",filename)))
        display_html(mpld3.display(fig))
        plt.close()
        # if len(classes) > 1:
        #     raise NotImplementedError()
        
        # dm=obj.getObsConfused(namesEscape(classes[0]),namesEscape(preds[0]),lim=lim)
        # if len(mods) > 0:
            # dm=dm[mods_]

        # plotDigits(dm,lim=lim,elemsByRows=nbCols,reshape=T)
