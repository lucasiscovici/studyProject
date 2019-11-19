from . import Viz
import plotly.figure_factory as ff
# from studyPipe import df_
import plotly.colors as pcol
# import plotly.figure_factory as ff
from studyPipeGit import df_,X_
import numpy as np
from plotly import graph_objs as go
from ..utils import T,F,namesEscape, merge, zipl
import pandas as pd
from itertools import combinations
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import plotly.colors as pcol
import plotly.express as pe
# def from
from copy import copy
def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]
def secureSort(ll):
    kepp=[i for i in ll if isinstance(i,int) or isinstance(i,float)]
    notKeep=[i for i in ll if not isinstance(i,int) and  not isinstance(i,float)]
    # print(kepp,notKeep)
    # print(np.sort(kepp).tolist()+notKeep)
    return np.sort(kepp).tolist()+notKeep
def fromCodeToCat(codes,lev):
    return pd.Categorical(codes).rename_categories(lev).tolist()
def heatmap3D(c1,c2,c3,yy=None):
    d1=c1 if isinstance(c1,pd.Series) or isinstance(c1,np.ndarray) or isinstance(c1,list) else (yy.loc[:,c1] if isinstance(c1,str) else yy.iloc[:,c1] )
    d2=c2 if isinstance(c2,pd.Series) or isinstance(c2,np.ndarray) or isinstance(c2,list) else (yy.loc[:,c2] if isinstance(c2,str) else yy.iloc[:,c2] )
    d3=c3 if isinstance(c3,pd.Series) or isinstance(c3,np.ndarray) or isinstance(c3,list) else (yy.loc[:,c3] if isinstance(c3,str) else yy.iloc[:,c3] )
    th=pd.crosstab([d1,d2],d3)
    nbCol=len(th.columns)
    nbRow=len(th.index.codes[0])
    return pd.DataFrame(dict(x=np.repeat(namesEscape(
        fromCodeToCat(th.index.codes[0],th.index.levels[0])),nbCol),mode="markers",
                                y=np.repeat(namesEscape(fromCodeToCat(th.index.codes[1],th.index.levels[1])),nbCol),
                                z=np.tile(namesEscape(th.columns.tolist()),nbRow),col=th.values.flatten()))
def plotHeatmap3D(d1,d2,d3):
    yu=heatmap3D(d1,d2,d3)
    return pe.scatter_3d(yu,x="x",y="y",z="z",color="col",color_continuous_scale="BlueRed") |_fun_| go.Figure
from copy import copy
def getNbRowsCols(datasl,nbCols=4):
    images_per_row = min(datasl,nbCols)
    n_rows = (datasl - 1) // images_per_row + 1
    return (n_rows,images_per_row)

def getLayoutsScene(l,offset,nb):
    yy=[]
    for i in range(offset,offset+nb):
        axn="" if i==0 else str(i+1)
        scn=l["scene"+axn]
        yy.append(scn)
    return yy

def fromCombiToplot(fig,combi,nbCols=4):
    d=fig.data
    l=fig.layout
    comb=np.array(combi)
    firstrow=len(np.unique(comb[:,0]))
    dfComb=pd.DataFrame(comb,columns=["f","s","t"])
    dfCombGr=dfComb.groupby(["f","s"])
    dfCombGrG=dfComb.groupby(["f"]).count()
    dfCombGrCount=dfCombGr.count()
    #offset=0
    nrowsG=0
    ncolsG=0
    dims=[]
    subT=[]
    iio=0
    for j in range(firstrow):
        nbJ=dfCombGrG.loc[j]["s"]
        #datasJ=d[offset:(offset+nbJ)]
        #layoutJ=getLayoutsScene(l,offset,nbJ)
        dfCombGrCountJ=dfCombGrCount.loc[j]
        nbT=dfCombGrCountJ.values.ravel()
        yba=[]
        for i_,j2 in enumerate(nbT):
            if i_ ==0:
                tb=[ "z : "+l["scene"+("" if i+iio==0 else str(i+iio+1))]["zaxis"]["title"]["text"] for i in range(j2)]
                #print([""]*(i_+j)+tb+[""]*(nbCols-j2-i_-j))
                yba.append([""]*(i_+j)+tb+[""]*(nbCols-j2-i_-j))
            else:
                yba.append([""]*j2+[""]*(nbCols-j2))
            nrows,ncols=getNbRowsCols(j2,nbCols)
            dims.append((nrows,ncols))
            nrowsG+=nrows
            ncolsG+=ncols
            iio+=j2
        #subTR=+[""]*(nbCols-j2)
        subT.append(yba)
    #print(nrowsG,nbCols)
    
        #offset+=nbJ
    #print(subT)
    f=make_subplots(rows=int(nrowsG),horizontal_spacing=0,vertical_spacing=0.05,cols=nbCols,specs=np.full((nrowsG,nbCols),[{'type': 'scene'}]).tolist(),subplot_titles=np.concatenate(subT).flatten())
    
    fl=f.layout
    offset=0
    nrowsOff=0
    datg=[]
    iio=0
    iioo=0
    layu={}
    #print(firstrow)
    for j in range(firstrow):
        nbJ=dfCombGrG.loc[j]["s"]
        #print("firstRow",j,nbJ)
        datasJ=d[(offset*2):(offset+nbJ)*2]
        
        layoutJ=getLayoutsScene(l,offset,nbJ)
        dfCombGrCountJ=dfCombGrCount.loc[j]
        nbT=dfCombGrCountJ.values.ravel()
        #print(nbT)
        iiop=0
        for i_,j2 in enumerate(nbT):
            nrows,ncols=dims[i_]
            iio2=0
            iio2X=0
            #print(iiop,j2)
            layoutJ2=layoutJ[iiop:(iiop+j2)]
            datasJ2=datasJ[iiop*2:(iiop+j2)*2]
            if i_==0:
                lsr=list(fl.annotations)
                ooi2=-0.1
                lop="scene"+("" if iio==0 else str(iio+1))
                ooi=fl[lop]["domain"]["y"][1]
                texto="x : "+layoutJ2[0]["xaxis"]["title"]["text"]
                fl.annotations=lsr+[
                        dict(yref="paper",
                        showarrow=F,
                        x=ooi2,
                        y=ooi,
                        font=dict(size=16),
                        xref="paper",
                        text=texto,
                        yanchor="top",
                        xanchor="center",
                        textangle=0)
                    ]
            #print("second",j2,nrows,ncols)
            for k in range(nrows):
                texto="y : "+layoutJ2[k]["yaxis"]["title"]["text"]
                #lop="scene"+("" if iio==0 else str(iio+1))
                for k2 in range(ncols):
                    #print("IIO",iio+1)
                    lop="scene"+("" if iio==0 else str(iio+1))
                    lopX="scene"+("" if iio+i_+j==0 else str(iio+i_+j+1))
                    #print("t",k,k2,layoutJ2,iio2)
                    layU=copy(layoutJ2[iio2])
                    layU2=fl[lopX]["domain"]
                    layU["domain"]=layU2
                    #print("lop",k,lop,lopX,layU2)
                    layU.camera.eye=dict(x=-1.5,y=1.5,z=-0.1)
                    layU["xaxis"]["title"]["text"]="x"
                    layU["yaxis"]["title"]["text"]="y"
                    layU["zaxis"]["title"]["text"]="z"
                    layu[lopX]=layU.to_plotly_json()
                    datU=datasJ2[iio2X]
                    datU2=datasJ2[iio2X+1]
                    
                    datU["scene"]=lopX
                    datU2["scene"]=lopX
                    datg.append(datU)
                    datg.append(datU2)
                    iio2+=1
                    iiop+=1
                    iioo+=1
                    iio+=1
                    iio2X+=2
                    if iio2 == (j2):
                        iio+=(nrows*nbCols-j2)
                        break
                lsr=list(fl.annotations)
                ooi2=max(layU2["x"])+0.1
                ooi=fl[lop]["domain"]["y"][1]
                #texto="x : "+layoutJ2[k]["yaxis"]["title"]["text"]
                fl.annotations=lsr+[
                        dict(yref="paper",
                        showarrow=F,
                        x=ooi2,
                        y=ooi,
                        font=dict(size=16),
                        xref="paper",
                        text=texto,
                        yanchor="top",
                        xanchor="center",
                        textangle=0)
                    ]
                nrowsOff+=1
                    
        offset+=nbJ
    layu["coloraxis"]=l["coloraxis"]
    #layu["coloraxis"]["colorbar"].titleside="right"
    layu["coloraxis"]["colorbar"]["x"]=1.25
    layu["showlegend"]=False
    return go.Figure(data=datg,layout=merge(fl.to_plotly_json(),layu,add=F))
def scatter3D_MESH(*,x,y,z,col,cmax=100,cmin=0,scat=None,onlyMesh=False,visible=True):
    data=([go.Scatter3d( marker=dict(color=col),x=x,y=y,
                       z=z,mode="markers")]if scat is None else [scat]) if not onlyMesh else [] 
    return go.Figure(data=data+[go.Mesh3d(cmax=cmax,cmin=cmin,colorscale="BlueRed",showscale=False,x=x,y=y,z=z,intensity=col,hoverinfo="none",visible=visible)])
class Study_Tuned_Viz(Viz):
    
    def plot_resultats(self,col=pcol.sequential.Bluered,d3=False,share_xaxes=True,
        share_yaxes=True,hideUpper=False,zmaxHisto=4,offset=0,iloc=slice(None),loc=slice(None),nb=None,max3d=15,begin3d=0,offset3d=0,
        scatterColorBarX=-0.2,mesh3d=True):
        obj=self.obj
        resu=obj.resultat
        yy=resu >>  df_.select(df_.starts_with("param_"),"mean_test_score")
        yy["mean_test_score"]=np.round(yy["mean_test_score"]*100,2)
        nb = nb if nb is not None else np.shape(yy)[1]-1
        nbo=min(nb+offset,nb)
        yyX=yy.iloc[:,:-1].iloc[:,offset:nbo].loc[:,loc].iloc[:,iloc]
        yyX["mean_test_score"]=yy["mean_test_score"]
        yy=yyX
        yy_=yy.fillna("`None`")
        yl=yy["mean_test_score"]
        cmax=max(yl)
        cmin=min(yl)

        # yy=yy.fillna("`None`")
        yy=yy_.apply(lambda x:namesEscape(x.values),axis=0)
        # print(yy)
        yy["mean_test_score"]=yl
        yy2=yy
        # print(yy)
        nb=np.shape(yy)[1]
        nb1=nb-1
        nbCols=nb1
        ff2=yy.columns.tolist()[:-1]
        if d3:
            nbCols-=2
            if nbCols==0:
                raise Exception("if 3D -> minimun 3 params")
            comb=list(combinations(range(nb1),3))[(begin3d + offset3d):(max3d+offset3d)]
            # Initialize figure with 4 3D subplots
            images_per_row = min(len(comb),nbCols)
            n_rows = (len(comb) - 1) // images_per_row + 1

            # print(comb)
            # print(n_rows)
            # print(images_per_row)
            spco=np.full((n_rows,images_per_row),[{'type': 'scatter3d'}]).tolist()
            fig = make_subplots(vertical_spacing=0.01,horizontal_spacing=0.01,
                rows=n_rows, cols=images_per_row,
                specs=spco)
            iio=0
            lay={}
            ip=None
            for i in range(n_rows):
                for j in range(images_per_row):
                    # print(i+1,j+1,iio)
                    com1=comb[iio]
                    x=ff2[com1[0]]
                    y=ff2[com1[1]]
                    z=ff2[com1[2]]
                    # print(x,y,z)
                    opX=px.scatter_3d(yy_,x=x, color_continuous_scale="Bluered", y=y, z=z,color="mean_test_score")
                    oip=opX["layout"]
                    xxo="scene"+("" if iio ==0 else str(iio+1))
                    oip[xxo]=oip["scene"]
                    if iio!=0:
                        oip["scene"]=None
                    opX["data"][0]["scene"]="scene"+("" if iio ==0 else str(iio+1))
                    opX['data'][0]["marker"].cmax=cmax
                    opX['data'][0]["marker"].cmin=cmin
                    opX["data"][0].hoverlabel=dict(bgcolor=yy_["mean_test_score"])
                    # opX["data"][0]["yaxis"]="y"+("" if i ==0 else str(i+1))
                    oip[xxo]["domain"]=None
                    if iio==0:
                        ip=oip["coloraxis"]
                    # opX["data"][0]["marker"]["color"]=pcol.sequential.Bluered
                    lay[xxo]=oip[xxo].to_plotly_json()
                    lay[xxo]["aspectmode"]= "cube"
                    fig.append_trace(
                             opX["data"][0],
                             row=i+1,col=j+1
                        )
                    fig.append_trace(
                             scatter3D_MESH(x=yy_.loc[:,x],y=yy_.loc[:,y],z=yy_.loc[:,z],cmax=cmax,cmin=cmin,col=yy_.loc[:,"mean_test_score"],onlyMesh=True,visible=mesh3d).data[0],
                             row=i+1,col=j+1
                        )
                    iio+=1
                    if len(comb) == iio:
                        lay["coloraxis"]=ip
                        fig["layout"]= merge(fig["layout"].to_plotly_json(),lay,add=False)
                        return fromCombiToplot(fig.update_layout(showlegend=False),comb,nbCols)
            # return fig
            # raise NotImplementedError() 
        else:
            # print(yy)

            oo2=ff.create_scatterplotmatrix(yy,index="mean_test_score",
                                    diag='histogram',marker=dict(colorbar=dict(x=scatterColorBarX,ticktextside="left",title="mean_test_score"),showscale=True),
                                    colormap=col,
                                    width=1000,
                                    height=1000,hovertemplate="<b>%{marker.color}%</b><br>" +
                "x : %{x}<br>"+
                "y : %{y}<br>"+"<extra></extra>",size=17).update_yaxes(automargin=True).update_xaxes(automargin=True)
            ii=0
            oo=oo2.data
            # print()
            diago=0

            od2=list(oo2.data)
            gh=True
            gh2=True
            # df2=
            mmax=0
            mmin=0
            for i in range(1,nb):
                for k in range(1,nb):
                    if i<k:
                        xx=yy_.iloc[:,k-1]
                        dt=od2[ii]
                        yy=dt["y"]
                        yy=yy_.iloc[:,i-1]
                        mmax_=np.max(pd.crosstab(np.array(yy),np.array(xx)).values)
                        mmin_=np.min(pd.crosstab(np.array(yy),np.array(xx)).values)
                        mmax=max(mmax,mmax_)
                        mmin=min(mmin,mmin_)
                        ii+=1
            ii=0
            for i in range(1,nb):
                i1=i
                x_ticker = ff2[i-1]
                for k in range(1,nb):
                    y_ticker = ff2[k-1]
                    k1=k
                    dt=od2[ii]
                    xx=dt["x"]
                    if i1<k1:
                        # print(ii)
                        # dt.update(visible=False)
                        xx=yy_.iloc[:,k-1]
                        yy=dt["y"]
                        # yy=yy.fillna("None")
                        yy=yy_.iloc[:,i-1]
                        ft=dt.xaxis
                        visible=True
                        if hideUpper:
                            ft="x2"
                            visible=False
                        zz=pd.crosstab(np.array(yy),np.array(xx))


                        figi=go.Heatmap(#blueRed
                            colorscale="oranges",showscale=gh,colorbar=dict(x=1.10,title="Count"),zmin=mmin,zmax=mmax,
                            z=zz.values,y=namesEscape(zz.index.tolist()),x=namesEscape(zz.columns.tolist()),xgap=2,ygap=2,xaxis=ft,yaxis=dt.yaxis,visible=visible)
                        gh=False
                        # figi.hoverlabel=dict(bgcolor=zz.values.tolist())
                        # if i1==1:
                            # oo2.layout["xaxis{}".format("" if ii==0 else ii+1)].tickangle=0
                        # if k1 == (nb-1):
                        #     oo2.layout["yaxis{}".format("" if ii==0 else ii+1)].tickvals=namesEscape(zz.index.tolist())
                        # print(ii)
                        # print(od2[ii])
                        # figi.name='{0} vs {1}'.format(x_ticker,y_ticker)
                        od2[ii]=figi

                    if i1 > k1 and gh2:
                        gh2=False
                        od2[ii].marker.showscale=True


                    if i1 == k1 :
                        zr=yy_.iloc[:,k-1]
                        # print(zr)
                        od2[ii].x = od2[ii].x if isinstance(zr[0],str) else namesEscape(secureSort(zr))
                        uni=unique(od2[ii].x)
                        # print(yy)
                        vl=yy2.groupby(x_ticker).mean().loc[uni,:].values.flatten().tolist()
                        # print(vl)
                        od2[ii].marker.cmin=cmin
                        od2[ii].marker.cmax=cmax
                        od2[ii].marker.color=vl
                        od2[ii].marker.colorscale="Bluered"
                        od2[ii]["hovertemplate"]="x ["+y_ticker+"]: %{x}<br>" +\
                                                "nb : %{y}<br>" +\
                                                "mean(mean_test_score) : %{marker.color} <br>" +\
                                                "<extra></extra>"
                    od2[ii]["name"]=""
                    if i1 < k1:
                        od2[ii]["hovertemplate"]="x ["+y_ticker+"]: %{x}<br>" +\
                                                "y ["+x_ticker+"]: %{y}<br>" +\
                                                "nb : %{z} <br>" +\
                                                "<extra></extra>"
                    if i1 > k1 :
                        od2[ii]["marker"]["opacity"]=0.5,
                        # zr=yy_.iloc[:,k-1]
                        # # print(zr)
                        # # print(isinstance(zr[0],str))
                        # od2[ii].x = od2[ii].x# if isinstance(zr[0],str) else namesEscape(secureSort(zr))
                        # # print(od2[ii].x)
                        # zr=yy_.iloc[:,i-1]
                        # # print(zr)
                        # od2[ii].y = od2[ii].y #if isinstance(zr[0],str) else namesEscape(secureSort(zr))
                        
                        xo=yy_.iloc[:,k-1]#od2[ii].x
                        yi=yy_.iloc[:,i-1]#od2[ii].y
                        colr=od2[ii].marker.color
                        bhh=pd.DataFrame(dict(a=xo,b=yi,z=colr))
                        # print(bhh)
                        bhh2_=bhh.groupby(["a","b"], sort=False)
                        bhh2=bhh2_.mean().reset_index()
                        bhj_=(bhh2_.count().values).flatten()
                        bhj=[min(10,int(i/2.)) for i in bhj_]
                        # print(bhj)
                        # raise Exception()
                        od2[ii].x=bhh2.loc[:,"a"]
                        od2[ii].y=bhh2.loc[:,"b"]
                        od2[ii].marker.color=bhh2.loc[:,"z"]
                        od2[ii].marker.line=dict(width=bhj,
                                        color='black')
                        od2[ii].customdata=bhj_
                        od2[ii]["hovertemplate"]="x ["+y_ticker+"]: %{x}<br>" +\
                                                "y ["+x_ticker+"]: %{y}<br>" +\
                                                "mean(mean_test_score) : %{marker.color}%<br>nb : %{customdata}" +\
                                                "<extra></extra>"

                        kkop=od2[ii].x
                        kkop2=od2[ii].y
                        od2[ii].x=namesEscape(od2[ii].x)
                        od2[ii].y=namesEscape(od2[ii].y)
                        # print(od2[ii].x)

                    ii+=1
                    if i1 > k1 :
                        zr=kkop.tolist()#yy_.iloc[:,k-1]
                        # print(zr)
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii+1)].tickangle=0
                        yj=kkop if isinstance(zr[0],str) else namesEscape(secureSort(zr))
                        # print(unique(yj))
                        oo2.layout["xaxis{}".format("" if ii==0 else ii)].categoryorder="array"
                        oo2.layout["xaxis{}".format("" if ii==0 else ii)].categoryarray=unique(yj).tolist()
                        oo2.layout["xaxis{}".format("" if ii==0 else ii)].type="category"
                        oo2.layout["xaxis{}".format("" if ii==0 else ii)].tickvals=unique(yj).tolist()
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].rangemode = 'tozero'
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].range= [0, len(unique(yj).tolist())-1]
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].tick0= ''
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].tickson='boundaries'#=unique(yj).tolist()
                        # yy=dt["y"]
                        zr=kkop2.tolist()
                        # print(type(zr[0]))
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii+1)].tickangle=0
                        yj2=zr if isinstance(zr[0],str) else namesEscape(secureSort(zr))
                        # print(unique(yj2))
                        # print(yj2)
                        # oo2.layout["yaxis{}".format("" if ii==0 else ii)].tickvals=unique(yj2).tolist()
                        # oo2.layout["yaxis{}".format("" if ii==0 else ii)].ticktext=unique(yj2).tolist()
                        oo2.layout["yaxis{}".format("" if ii==0 else ii)].categoryorder="array"
                        oo2.layout["yaxis{}".format("" if ii==0 else ii)].categoryarray=unique(yj2).tolist()
                        oo2.layout["yaxis{}".format("" if ii==0 else ii)].type="category"
                        oo2.layout["yaxis{}".format("" if ii==0 else ii)].tickvals=unique(yj2).tolist()
                        oo2.layout["yaxis{}".format("" if ii==0 else ii)].rangemode = 'tozero'
                        # oo2.layout["yaxis{}".format("" if ii==0 else ii)].range= [0, len(unique(yj2).tolist())-1]                         
                        # gg=list(oo2.data)
                    
                    if i1<k1 and hideUpper:
                         oo2.layout["xaxis{}".format("" if ii==0 else ii)].ticks=""
                         oo2.layout["xaxis{}".format("" if ii==0 else ii)].tickvals=[]
                         oo2.layout["xaxis{}".format("" if ii==0 else ii)].showgrid=False
                         oo2.layout["xaxis{}".format("" if ii==0 else ii)].zeroline=False
                         oo2.layout["yaxis{}".format("" if ii==0 else ii)].ticks=""
                         oo2.layout["yaxis{}".format("" if ii==0 else ii)].tickvals=[]
                         oo2.layout["yaxis{}".format("" if ii==0 else ii)].showgrid=False
                         oo2.layout["yaxis{}".format("" if ii==0 else ii)].zeroline=False
                
                    ls=list(oo2.layout.annotations)
                    if k1 == i1:
                        u=dt
                        xx="x{}".format("" if ii==0 else ii)
                        yy="y{}".format("" if ii==0 else ii)
                        ooo=oo2.layout["yaxis{}".format("" if ii==0 else ii)]
                        ooo2=oo2.layout["xaxis{}".format("" if ii==0 else ii)]
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].tickangle=0
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
                        # if k1 >1:
                        oo2.layout["xaxis{}".format("" if ii==0 else ii)].showticklabels=F
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].tickangle=45
                        # oo2.layout["yaxis{}".format("" if ii==0 else ii)].showticklabels=F
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].showgrid=True
                    # if i1==2 and k1==1:
                        # oo2.layout["xaxis{}".format("" if ii==0 else ii)].showticklabels=T
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