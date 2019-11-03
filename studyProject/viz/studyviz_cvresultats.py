import plotly.figure_factory as ff
import numpy as np
from ..utils import StudyClass, F, T
from . import Viz

class Study_CvResultats_Viz(Viz):
	def plot_confusion_matrix(self,y_true,namesY=None,normalize=True,addDiagonale=True,colorscale="RdBu",
								showscale=True,reversescale=True,size=18,width=500,line_color="red",line_dash="longdash",
								line_width=6,border=True,xlabel="Predict",ylabel="Actuelle",addCount=True,name="Diag",title=None,plots_kwargs={},me=None):
		obj=self.obj
		# print(y_true)
		if me is not None:
			if isinstance(y_true,str):
				y_true=getattr(me,y_true)
			if isinstance(namesY,str):
				namesY=getattr(me,namesY).cat
		confMatCls=obj.confusion_matrix(y_true,namesY=namesY,normalize=normalize,returnNamesY=True)
		confMat=confMatCls.confusion_matrix
		confMatNamesY=confMatCls.namesY
		zmax=np.max(confMat.values)
		zmin=np.min(confMat.values)
		vla=confMat.values
		annotation_text=None
		if addCount and normalize:
			vlaS=vla.shape
			confMat2=obj.confusion_matrix(y_true,namesY=namesY,normalize=False,returnNamesY=False)
			fe=np.repeat(np.sum(confMat2.values,axis=1),vlaS[0])
			# print(fe)
			annotation_text=list(map(lambda a: "{}%<br />{}/{}".format(a[1][0],a[1][1],fe[a[0]]),enumerate(zip(vla.flatten(),confMat2.values.flatten()))))
			annotation_text=np.reshape(annotation_text,vlaS)

		conf=ff.create_annotated_heatmap(vla,annotation_text=annotation_text,
									zmid=(zmax-zmin)/2.,
                            x=confMatNamesY,y=confMatNamesY,
                            colorscale=colorscale,showscale=showscale,reversescale=reversescale,**plots_kwargs).update_layout(
                                                                                                font=dict(size=size),
                                                                         width=width)
		if addDiagonale:
			conf=conf.add_scatter(x=confMatNamesY,
									y=confMatNamesY[::-1],
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

		# if border:
		# 	conf.update_layout(
		# 		xaxis=dict(linecolor = "black",linewidth=5))
		# 	conf.update_layout(
		# 		yaxis=dict(linecolor = "black"))
		return conf