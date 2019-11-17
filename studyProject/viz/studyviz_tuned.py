from . import Viz
import plotly.figure_factory as ff
# from studyPipe import df_
import plotly.colors as pcol
# import plotly.figure_factory as ff
from studyPipeGit import df_,X_
import numpy as np
from plotly import graph_objs as go
from ..utils import T,F
class Study_Tuned_Viz(Viz):
    
    def plot_resultats(self,col=pcol.sequential.Bluered,d3=False,share_xaxes=True,
        share_yaxes=True):
        obj=self.obj
        resu=obj.resultat
        yy=resu >>  df_.select(df_.starts_with("param_"),"mean_test_score")
        yy["mean_test_score"]=np.round(yy["mean_test_score"]*100,2)
        nb=np.shape(yy)[1]
        ff2=yy.columns.tolist()[:-1]
        if d3:
            raise NotImplementedError() 
        else:
            oo2=ff.create_scatterplotmatrix(yy,index="mean_test_score",
                                    diag='histogram',marker=dict(colorbar=dict(x=-0.15,title="Scatter"),showscale=True),
                                    colormap=col,
                                    width=1000,
                                    height=1000,hovertemplate="<b>%{marker.color}%</b><br>" +
                "x : %{x}<br>"+
                "y : %{y}<br>"+"<extra></extra>",size=17,).update_yaxes(automargin=True).update_xaxes(automargin=True)
            ii=0
            oo=oo2.data
            # print()
            diago=0

            od2=list(oo2.data)
            gh=True
            gh2=True
            for i in range(1,nb):
                i1=i
                for k in range(1,nb):
                    k1=k
                    dt=od2[ii]
                    if i1<k1:
                        # print(ii)
                        # dt.update(visible=False)
                        xx=dt["x"]
                        yy=dt["y"]
                        figi=go.Histogram2d(#blueRed
                            colorscale="oranges",showscale=gh,colorbar=dict(x=1.15,title="Histogram2d"),zmin=0,zmax=4,
                            x=xx,xgap=2,ygap=2,xaxis=dt.xaxis,yaxis=dt.yaxis,
                            y=yy)
                        gh=False
                        # print(ii)
                        # print(od2[ii])
                        od2[ii]=figi
                    if i1 > k1 and gh2:
                        gh2=False
                        od2[ii].marker.showscale=True
                    if i1 == k1 :
                        od2[ii].marker.color="black"
                    ii+=1
                    # if i1 < k1 :
                    #     oo2.layout["xaxis{}".format("" if ii==0 else ii)].tickvals=np.unique(xx)
                    #     oo2.layout["yaxis{}".format("" if ii==0 else ii)].tickvals=np.unique(yy)                            
                        # gg=list(oo2.data)
                    
                    # if i1<k1:
                         # oo2.layout["xaxis{}".format("" if ii==0 else ii)].ticks=""
                         # oo2.layout["xaxis{}".format("" if ii==0 else ii)].tickvals=[]
                         # oo2.layout["xaxis{}".format("" if ii==0 else ii)].showgrid=False
                         # oo2.layout["xaxis{}".format("" if ii==0 else ii)].zeroline=False
                         # oo2.layout["yaxis{}".format("" if ii==0 else ii)].ticks=""
                         # oo2.layout["yaxis{}".format("" if ii==0 else ii)].tickvals=[]
                         # oo2.layout["yaxis{}".format("" if ii==0 else ii)].showgrid=False
                         # oo2.layout["yaxis{}".format("" if ii==0 else ii)].zeroline=False
                
                    ls=list(oo2.layout.annotations)
                    if k1 == i1:
                        u=dt
                        xx="x{}".format("" if ii==0 else ii)
                        yy="y{}".format("" if ii==0 else ii)
                        ooo=oo2.layout["yaxis{}".format("" if ii==0 else ii)]
                        ooo2=oo2.layout["xaxis{}".format("" if ii==0 else ii)]
                        oo2.layout["xaxis{}".format("" if ii==0 else ii)].tickangle=0
                        # print(ooo.domain)
                        texto=ff2[diago]
                        diago+=1
                        ooi=dt["y"]
                        ooi=max(ooo.domain)+0.01
                        ooi2=sum(ooo2.domain)/2.
                        # print(ooo.domain)
                        # print(max(ooo.domain)-min(ooo.domain))
                        oo2.layout.annotations=ls+[
                            dict(yref="paper",
                            showarrow=F,
                            x=ooi2,
                            y= ooi,
                            xref="paper",
                            text=texto,
                            yanchor="bottom",
                            xanchor="center",
                            textangle=0)
                        ]
                        # oo2.data[ii].title.text = ff2[]

                    if k1==1 or i1==(nb-1):
                        # print(ii)
                        oo2.layout["xaxis{}".format("" if ii==0 else ii)].title.text=""
                        oo2.layout["yaxis{}".format("" if ii==0 else ii)].title.text=""
                    if share_xaxes and i1>1 and i1<(nb-1) and i1 != k1:
                        # print(ii)
                        oo2.layout["xaxis{}".format("" if ii==0 else ii)].showticklabels=F
                        # oo2.layout["yaxis{}".format("" if ii==0 else ii)].showticklabels=F
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].showgrid=True
                    if share_yaxes and (i1==k1 or k1 == (nb-1)):
                        oo2.layout["yaxis{}".format("" if ii==0 else ii)].side="right"
                    if share_yaxes and k1>1 and i1 != k1 and k1 != (nb-1): 
                        oo2.layout["yaxis{}".format("" if ii==0 else ii)].showticklabels=F

                    # if share_yaxes and i1==(k1+1) and k1 > 2 and not oo2.layout["yaxis{}".format("" if ii==0 else ii)].showticklabels:
                    #     oo2.layout["yaxis{}".format("" if ii==0 else ii)].side="right"
                    #     oo2.layout["yaxis{}".format("" if ii==0 else ii)].showticklabels=T

        oo2=go.Figure(data=od2,layout=oo2.layout)
        return oo2

# data=[go.Scatter3d(
#                        marker=dict(color=yy.mean_test_score.values.tolist()),x=yy.param_max_depth,y=yy.param_max_features,
#                        z=yy.param_min_samples_leaf,mode="markers")]
# go.Figure(data=data+[go.Mesh3d(x=yy.param_max_depth,y=yy.param_max_features,z=yy.param_min_samples_leaf,intensity=yy.mean_test_score.values.tolist())])
# import plotly.express as px
# px.scatter_3d(yy,x="param_n_estimators",y="param_min_samples_split",z="param_min_samples_leaf",
#          color="mean_test_score",
#          color_continuous_scale="BlueRed")