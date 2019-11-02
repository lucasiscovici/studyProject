import plotly.figure_factory as ff
import numpy as np
from ..utils import StudyClass,namesEscape
from . import Viz
from plotly.subplots import make_subplots
import cufflinks as cf
from plotly.offline import iplot
import plotly.graph_objs as go
from functools import reduce
import operator
from ..utils import isStr
class Study_CrossValidItem_Viz(Viz):
	def plot_confusion_matrix(self,y_true,namesY=None,mods=[],normalize=True,addDiagonale=True,colorscale="RdBu",showscale=True,reversescale=True,size=18,width=500,line_color="red",line_dash="longdash",line_width=6,nbCols=3,colFixed=None,shared_xaxes=True,
								shared_yaxes=False,vertical_spacing=0.02,horizontal_spacing=0.15,title=None,plots_kwargs={}):
		obj=self.obj
		modsN=list(obj.resultats.keys())
		models=obj.resultats
		if len(mods)>0:
			mods_ = [i if isStr(i) else modsN[i] for i in mods]
			models= {i:obj.resultats[i] for i in mods_}

		namesY= namesEscape(namesY) if namesY is not None else namesY
		confMatCls={k:v.viz.plot_confusion_matrix(y_true,
				namesY=namesY,normalize=normalize,addDiagonale=addDiagonale,colorscale=colorscale,
				showscale=showscale,reversescale=reversescale,size=size,width=width,
				line_color=line_color,line_dash=line_dash,line_width=line_width,plots_kwargs=plots_kwargs,title="Confusion Matrix of {}".format(k),name="Diag {}".format(i_+1)) 
		for i_,(k,v) in enumerate(models.items())}
		nbCols=min(len(confMatCls),nbCols)
		images_per_row=nbCols
		images_per_row = min(len(confMatCls), images_per_row)
		n_rows = (len(confMatCls) - 1) // images_per_row + 1

		rowsCol=(len(confMatCls) if colFixed is not None else n_rows,colFixed if colFixed is not None else nbCols)
		titles=[]
		tiplesS=[i.layout.title.text  for i in confMatCls.values()]
		subpl=cf.subplots(list(confMatCls.values()),shape=rowsCol,shared_xaxes=shared_xaxes,shared_yaxes=shared_yaxes,
                           horizontal_spacing=horizontal_spacing,
                           vertical_spacing=vertical_spacing,subplot_titles=tiplesS)
		
		annot=[i.layout.annotations for i in confMatCls.values()]
	
		size=list(confMatCls.values())[0]["layout"]["font"]["size"]

		X=[("x{}".format(j+1),"y{}".format(i+1)) for i in range(rowsCol[0]) for j in range(rowsCol[1])]
		def modifAnnotRef(annot,xn,yn):
			k=[]
			for i in annot:
				i.xref=xn
				i.yref=yn
				k.append(i)
			return k
		annot2=reduce(operator.add,[modifAnnotRef(annot[i],x,y) for i,(x,y) in enumerate(X)])
		subpl["layout"]["annotations"]=subpl["layout"]["annotations"]+annot2
		subpl["layout"]["font"]=dict(size=size)

		subpl["layout"]

		fig= go.Figure(subpl)
		fig.update_layout(legend=dict(x=0, y=-0.1),legend_orientation="h")
		if title is None:
			fig.update_layout(title_text="Confusion Matrix : cv '{}'".format(obj.ID))
		return fig