from . import Viz
from interface import implements
from ..utils import merge,  T,F, dicoAuto, zipl, StudyClass
import pandas as pd
import numpy as np


# from ..utils import T,F
class Study_StudyClassif__Viz(Viz):
    # @staticmethod
    def plot_confusion_matrix(self,y_true="y_train",namesY="train_datas",mods=[],normalize=True,addDiagonale=True,colorscale="RdBu",showscale=True,reversescale=True,size=18,width=500,line_color="red",line_dash="longdash",line_width=6,
            nbCols=3,colFixed=None,shared_xaxes=True,
                                    shared_yaxes=False,vertical_spacing=0.02,horizontal_spacing=0.15,title=None,plots_kwargs={},
                                    modelsNames=None,cvName=None,prefixTitle="Confusion Matrix of ",me=None):
        obj=self.obj
        return obj.currCV.viz.plot_confusion_matrix(y_true,
                                    namesY,
                                    mods,
                                    normalize,
                                    addDiagonale,
                                    colorscale,
                                    showscale,
                                    reversescale,
                                    size,
                                    width,
                                    line_color,
                                    line_dash,
                                    line_width,
                                    nbCols,
                                    colFixed,
                                    shared_xaxes,
                                    shared_yaxes,
                                    vertical_spacing,
                                    horizontal_spacing,
                                    title,
                                    plots_kwargs,
                                    modelsNames,
                                    cvName,
                                    prefixTitle,
                                    me)

    def plot_classification_report(self,y_true="y_train",namesY="train_datas",mods=[],normalize=True,addDiagonale=True,colorscale="RdBu",showscale=True,reversescale=True,size=18,width=500,line_color="red",line_dash="longdash",line_width=6,
            nbCols=3,colFixed=None,shared_xaxes=True,show=True,
                                    shared_yaxes=False,vertical_spacing=0.02,horizontal_spacing=0.15,title=None,plots_kwargs={},
                                    modelsNames=None,cvName=None,prefixTitle="Confusion Matrix of ",me=None):
        obj=self.obj
        y=[]
        for k,v in obj.cv.items():
            y.append(v.viz.plot_classification_report(y_true,
                namesY,
            mods,
            normalize,
            addDiagonale,
            colorscale,
            showscale,
            reversescale,
            size,
            width,
            line_color,
            line_dash,
            line_width,
            nbCols,
            colFixed,
            shared_xaxes,
            shared_yaxes,
            vertical_spacing,
            horizontal_spacing,
            title,
            plots_kwargs,
            modelsNames,
            cvName,
            prefixTitle,
            me))
        if show:
            for k in y:
                k.show()
        return y