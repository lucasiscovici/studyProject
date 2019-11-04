import plotly.figure_factory as ff
import numpy as np
from ..utils import StudyClass, F, T, merge
from . import Viz

class Study_CvResultats_Viz(Viz):
	def plot_confusion_matrix(self,y_true="y_train",namesY="train_datas",normalize=True,addDiagonale=True,colorscale="RdBu",
								showscale=True,reversescale=True,size=18,width=500,zmax=None,zmin=None,zmid=None,line_color="red",line_dash="longdash",
								line_width=6,border=True,xlabel="Predict",ylabel="Actuelle",addCount=True,name="Diag",
								title=None,plots_kwargs={},chutDiag=True,dontRescale=False,onlyConfMat=False,noLabel=False,me=None):
		obj=self.obj
		# print(y_true)                               "train_datas"
		if me is not None:
			if isinstance(y_true,str):
				y_true=getattr(me,y_true)
			if isinstance(namesY,str):
				namesY=getattr(me,namesY).cat
		confMatCls=obj.confusion_matrix(y_true,namesY=namesY,normalize=normalize,returnNamesY=True)
		confMat=confMatCls.confusion_matrix
		confMatNamesY=confMatCls.namesY
		zmax=np.max(confMat.values) if zmax is None else zmax
		zmin=np.min(confMat.values) if zmin is None else zmin
		vla=confMat.values
		vlaS=vla.shape
		# print(vla)
		vlaae=np.copy(vla)
		annotation_text=list(map(lambda a: "{}%".format(np.round(a) if np.round(a,2)>0.0 or not noLabel else ""),vla.flatten()))
		annotation_text=np.reshape(annotation_text,vlaS)
		zmid=(zmax-zmin)/2. if zmid is None else zmid
		if addCount and normalize:
			confMat2=obj.confusion_matrix(y_true,namesY=namesY,normalize=False,returnNamesY=False)
			fe=np.repeat(np.sum(confMat2.values,axis=1),vlaS[0])
			# print(fe)
			annotation_text=list(map(lambda a: "{}%<br />{}/{}".format(np.round(a[1][0],2),a[1][1],fe[a[0]]) if np.round(a[1][0],2)>0.0 or not noLabel else "",enumerate(zip(vla.flatten(),confMat2.values.flatten()))))
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

		# if border:
		# 	conf.update_layout(
		# 		xaxis=dict(linecolor = "black",linewidth=5))
		# 	conf.update_layout(
		# 		yaxis=dict(linecolor = "black"))
		return conf


	def plot_classification_report(self,y_true="y_train",namesY="train_datas",
		colorscale="Greys",reversescale=True,showscale=True,round_val=2,
		paper_bgcol="#F5F6F9",plot_bgcolor="black",
		linecolor="black",linewidth=2,noLabel=False,
		xgap=1,ygap=1,
		me=None):
		obj=self.obj
		if me is not None:
			if isinstance(y_true,str):
				y_true=getattr(me,y_true)
			if isinstance(namesY,str):
				namesY=getattr(me,namesY).cat
		cr=obj.classification_report(y_true,namesY)
		vlaS=cr.values*100.
		annotation_text=list(map(lambda a: "{}%".format(np.round(a[1],round_val)) if np.round(a[1],round_val)>0.0 or not noLabel else "",
			enumerate(vlaS.flatten())))
		annotation_text=np.reshape(annotation_text,vlaS)
		dd=ff.create_annotated_heatmap(vlaS.round(round_val),x=ff.columns.tolist(),y=ff.index.tolist(),
			annotation_text=annotation_text	,showscale=showscale,colorscale=colorscale,reversescale=reversescale)
		dd.update_layout(paper_bgcolor= paper_bgcol,
	                      plot_bgcolor= plot_bgcolor,
	                        xaxis=dict(side="bottom",linewidth= linewidth,mirror=True,
	                        	showgrid=F,linecolor = linecolor,zeroline=F,showline=T),
	                        yaxis=dict(linewidth= linewidth,mirror=True,showgrid=F,linecolor = linecolor,
	                        	zeroline=F,showline=T))
		dd.data[0].update(dict(xgap=xgap,ygap =ygap))
		dd.update_layout(width=560,height=600)
		return dd
