import plotly.figure_factory as ff
import numpy as np
from ..utils import StudyClass, F, T, merge
from . import Viz
import matplotlib.pyplot as plt
from ..utils import flipScale, frontColorFromColorscaleAndValues, namesEscape, isArr
class Study_CvResultatsClassif_Viz(Viz):
    def plot_confusion_matrix(self,y_true="y_train",namesY="train_datas",normalize=True,addDiagonale=True,colorscale="RdBu",
                                showscale=True,reversescale=True,size=18,round_val=2,width=500,zmax=None,zmin=None,zmid=None,line_color="red",line_dash="longdash",
                                line_width=6,lim=None,border=True,relative=False,xlabel="Predict",ylabel="Actuelle",addCount=True,name="Diag",
                                title=None,axis=1,plots_kwargs={},chutDiag=True,dontRescale=False,onlyConfMat=False,noLabel=False,me=None):
        obj=self.obj
        # print(y_true)                               "train_datas"
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        elif isinstance(y_true,str) or isinstance(namesY,str):
            me=obj.papa.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat


        confMatCls=obj.confusion_matrix(y_true,namesY=namesY,relative=relative,sum_=False,normalize=normalize,returnNamesY=True,axis=axis)
        confMat=np.nan_to_num(confMatCls.confusion_matrix)
        confMatClsX=obj.confusion_matrix(y_true,namesY=namesY,relative=False,sum_=False,normalize=normalize,returnNamesY=True,axis=axis)
        confMatX=np.nan_to_num(confMatClsX.confusion_matrix)

        confMatNamesY=confMatCls.namesY
        zmax=np.max(confMat) if zmax is None else zmax
        zmin=np.min(confMat) if zmin is None else zmin
        vla=confMat
        vlaS=vla.shape
        # print(vla)
        vlaae=np.copy(vla)
        vlf=vla.flatten()
        ok=np.repeat(True,np.shape(vlf)[0])
        if lim is not None:
            i=np.full(np.shape(vla),False)
            np.fill_diagonal(i,True)
            i=i.flatten()
            xxx=vla.copy()
            np.fill_diagonal(xxx,0)
            xxx=xxx.flatten()
            i[np.argsort(xxx)[::-1][:lim]]=T
            ok=i
        annotation_text=list(map(lambda a: "{}".format(np.round(a[1],round_val) if np.round(a[1],round_val)>0.0 and not noLabel and ok[a[0]] else ""),enumerate(vla.flatten())))
        annotation_text=np.reshape(annotation_text,vlaS)
        zmid=(zmax-zmin)/2. if zmid is None else zmid
        if addCount and normalize:
            confMat2=np.nan_to_num(obj.confusion_matrix(y_true,namesY=namesY,relative=relative,sum_=False,normalize=False,returnNamesY=False,axis=axis))
            confMat2X=np.nan_to_num(obj.confusion_matrix(y_true,namesY=namesY,relative=False,sum_=False,normalize=False,returnNamesY=False,axis=axis))
            fe=np.sum(confMat2,axis=axis,keepdims=True)
            fe2=np.sum(confMat2X,axis=axis,keepdims=True)
            fe=np.tile(fe,vlaS[0]).flatten()
            fe2=np.tile(fe2,vlaS[0]).flatten()
            vlo2=vla
            if relative:
                vlo2=vla.copy()
                np.fill_diagonal(vlo2,confMatX.diagonal())
                np.fill_diagonal(confMat2,confMat2X.diagonal())
                fe=fe.reshape(vlaS)
                fe2=fe2.reshape(vlaS)
                np.fill_diagonal(fe,fe2.diagonal())
                # print(fe2.diagonal())
                fe=fe.flatten()
            if lim is not None:
                i=np.full(np.shape(vla),False)
                np.fill_diagonal(i,True)
                i=i.flatten()
                xxx=vlo2.copy()
                np.fill_diagonal(xxx,0)
                xxx=xxx.flatten()
                i[np.argsort(xxx)[::-1][:lim]]=T
                ok=i
            # print(fe)
            # print(confMat2)
            # print(vlo2)
            # print(ok)
            annotation_text=list(map(lambda a: "{}%<br />{}/{}".format(np.round(a[1][0],round_val),a[1][1],fe[a[0]]) if np.round(a[1][0],round_val)>0.0 and not noLabel and ok[a[0]]  else "",enumerate(zip(vlo2.flatten(),confMat2.flatten()))))
            annotation_text=np.reshape(annotation_text,vlaS)
        if chutDiag:
            vlo=vla
            np.fill_diagonal(vlo,0)
            vla=vlo
            zmid=None
            if not dontRescale:
                zmax=np.max(vla)
                zmin=np.min(vla)

        if onlyConfMat:
            return [zmin,zmax]
        # print(zmin,zmax,zmid)
        argus=dict(annotation_text=annotation_text,
                                    zmid=zmid,
                                    zmax=zmax,
                                    text=vlaae,
                                    zmin=zmin,
                            x=confMatNamesY,y=confMatNamesY,
                            colorscale=colorscale,showscale=showscale,reversescale=reversescale)
        
        argus2=merge(argus,plots_kwargs,add=False)
        conf=ff.create_annotated_heatmap(vla,**argus2).update_layout(font=dict(size=size),
                                                                         width=width)
        if addDiagonale:
            conf=conf.add_scatter(x=confMatNamesY,
                                    y=confMatNamesY,
                                    hoverinfo="none",
                                    hovertemplate ="",
                                    showlegend = False,
                                    line=dict(color=line_color,
                                        dash=line_dash, 
                                        width=line_width),name=name
                                    )
        conf=conf.update_layout(yaxis_title=ylabel,xaxis_title=xlabel).update_xaxes(side="bottom")#.add_annotation(text=xlabel,x=0.49,y=-0.15,font=dict(size=size),showarrow=F)
        if title is None:
            # print(title)
            conf=conf.update_layout(title_text="Confusion matrix {}".format('' if obj.name is None else obj.name))
        else:
            conf=conf.update_layout(title_text=title)


        conf.data[0].update(hovertemplate = "<b>%{text}%</b><br>" +
            xlabel+" : %{y}<br>"+
            ylabel+" : %{x}<br>"+"<extra></extra>")


        if normalize and not noLabel and not zmin is None or zmax is None:
            ooo=np.array(vla.flatten())/100.
            o=[]
            cl=conf.data[0].colorscale
            rcl=conf.data[0].reversescale
            cll=flipScale(cl) if rcl else cl
            clrs=frontColorFromColorscaleAndValues(ooo,cll,zmin=zmin/100.,zmax=zmax/100.)
            for c,i in zip(clrs,conf.layout.annotations):
                i.font.color=c

        # if border:
        #   conf.update_layout(
        #       xaxis=dict(linecolor = "black",linewidth=5))
        #   conf.update_layout(
        #       yaxis=dict(linecolor = "black"))
        return conf


    def plot_classification_report(self,y_true="y_train",namesY="train_datas",
        colorscale="Greys",reversescale=False,showscale=True,round_val=2,
        paper_bgcol="#F5F6F9",plot_bgcolor="black",zmax=None,zmin=None,
        linecolor="black",linewidth=2,title=None,noLabel=False,onlyConfMat=False,
        xgap=1,ygap=1,
        me=None,*args,**xargs):
        obj=self.obj
        if me is not None:
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        elif isinstance(y_true,str) or isinstance(namesY,str):
            me=obj.papa.papa
            if isinstance(y_true,str):
                y_true=getattr(me,y_true)
            if isinstance(namesY,str):
                namesY=getattr(me,namesY).cat
        cr=obj.classification_report(y_true,namesY)[::-1]
        vlaS=cr.values*100.
        if onlyConfMat:
            zmax=np.max(vlaS)
            zmin=np.min(vlaS)
            return [zmin,zmax]


        annotation_text=list(map(lambda a: "{}%".format(np.round(a[1],round_val)) if np.round(a[1],round_val)>0.0 and not noLabel else "",
            enumerate(vlaS.flatten())))
        vlaSe=vlaS.shape
        # print(annotation_text)
        # print(vlaS)
        annotation_text=np.reshape(annotation_text,vlaSe)
        dd=ff.create_annotated_heatmap(vlaS.round(round_val),zmax=zmax,zmin=zmin,x=cr.columns.tolist(),y=cr.index.tolist(),
            annotation_text=annotation_text ,showscale=showscale,colorscale=colorscale,reversescale=reversescale)
        dd.update_layout(paper_bgcolor= paper_bgcol,
                          plot_bgcolor= plot_bgcolor,
                            xaxis=dict(side="bottom",linewidth= linewidth,mirror=True,
                                showgrid=F,linecolor = linecolor,zeroline=F,showline=T),
                            yaxis=dict(linewidth= linewidth,mirror=True,showgrid=F,linecolor = linecolor,
                                zeroline=F,showline=T))
        dd.data[0].update(dict(xgap=xgap,ygap =ygap))
        dd.update_layout(width=560,height=600)
        dd=dd.update_layout(title_text="Classification report {}".format('' if obj.name is None else obj.name) if title is None else title)
        
        if not noLabel and not zmin is None and not zmax is None:
            ooo=np.array(vlaS.flatten())/100.
            o=[]
            cl=dd.data[0].colorscale
            rcl=dd.data[0].reversescale
            cll=flipScale(cl) if rcl else cl
            clrs=frontColorFromColorscaleAndValues(ooo,cll,zmin=zmin/100.,zmax=zmax/100.)
            for c,i in zip(clrs,dd.layout.annotations):
                i.font.color=c
        return dd
