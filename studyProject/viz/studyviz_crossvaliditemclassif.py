import plotly.figure_factory as ff
import numpy as np
from ..utils import StudyClass,namesEscape
from . import Viz
from plotly.subplots import make_subplots
import cufflinks as cf
from cufflinks.tools import get_len
from plotly.offline import iplot
import plotly.graph_objs as go
from functools import reduce
import operator
from ..utils import isStr
from operator import itemgetter

class Study_CrossValidItem_Viz(Viz):
	def plot_confusion_matrix(self,y_true="y_train",namesY="train_datas",mods=[],normalize=True,addDiagonale=True,colorscale="RdBu",
		showscale=True,reversescale=True,size=18,width=500,line_color="red",line_dash="longdash",line_width=6,
		nbCols=3,colFixed=None,shared_xaxes=True,
								shared_yaxes=False,vertical_spacing=0.02,horizontal_spacing=0.15,title=None,plots_kwargs={},
								modelsNames=None,cvName=None,prefixTitle="Confusion Matrix of ",me=None,**plotConfMat_kwargs):
		# print(y_true)
		if me is not None:
			if isinstance(y_true,str):
				y_true=getattr(me,y_true)
			if isinstance(namesY,str):
				namesY=getattr(me,namesY).cat
		obj=self.obj
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
			confMatM=[v.viz.plot_confusion_matrix(y_true,
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
		confMatCls={k:v.viz.plot_confusion_matrix(y_true,
				namesY=namesY,normalize=normalize,addDiagonale=addDiagonale,colorscale=colorscale,
				showscale=showscale,reversescale=reversescale,size=size,width=width,
				line_color=line_color,line_dash=line_dash,line_width=line_width,plots_kwargs=plots_kwargs,title=(prefixTitle+"{}").format(k),name="Diag {}".format(i_+1),**plotConfMat_kwargs) 
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
			i.update(xaxis_title="",yaxis_title="")
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
			fig.update_layout(title_text="Confusion Matrix : cv '{}'".format(obj.ID if cvName is None else cvName))
		
		# for axis, n in list(get_len(fig).items()):
		# 	for u in range(1,n+1):
		# 		_='' if u==1 else u
		# 		o=0 if axis == "x" else 1
		# 		if shared_xaxes and axis=="x" and u > 1:
		# 			continue
		# 		fig['layout']['{0}axis{1}'.format(axis,_)]["title"]=dict(text=tiplesSAxis[u-1][o])
		
		fig.update_layout(legend=dict(x=0, y=-0.1),
						  legend_orientation="h")

		# print(list(confMatCls.values())[0]["data"])
		for i in fig.data:
			if i.__class__.__name__=="Heatmap":
				i.update(hovertemplate = "<b>%{text}%</b><br>" +
				tiplesSAxis[0][1]+" : %{y}<br>" +
				tiplesSAxis[0][0]+" : %{x}<br>" + "<extra></extra>")
		fig.update_layout(xaxis_title=tiplesSAxis[0][0],yaxis_title=tiplesSAxis[0][1])

		# fig.update_xaxis(title_text=tiplesSAxis[0])
		# fig.update_yaxis(title_text=tiplesSAxis[1])

		return fig