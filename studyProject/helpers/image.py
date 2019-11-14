import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# def plotImgs(im,title="",nr=2,nc=5,figsize=(9,5),w=28,h=28,titleSize=29,reshape=True,*args,**xargs):
#     uu=reshape(im,w,h) if reshape else im
#     plt.figure(figsize=figsize)
#     for _i,i in enumerate(uu): 
#         plt.subplot(nr,nc,_i+1)
#         plt.imshow(i)
#         plt.axis('off')
#     plt.suptitle(title,size=titleSize);

def reshapeMultiClassif(im,w=28,h=28):
    return [reshape(j,w,h) for ww,j in im.items()]
    
def reshape(im,w=28,h=28):
    return [i.reshape(w,h) for i in im]
    
def plotImgs(instances,title="", elemsByRows=10,w=28,h=28,figsize=None,
                titleSize=29,reshape=True,lim=100,show=True,returnOK=False,noImg=True,ax=None,filename="img.png",beforeImgAx=lambda ax:None,multiple=False,maxMultiple=None, **options):
    from ..utils import IMG, ifelse
    import mpld3
    from IPython.display import display_html, HTML, FileLink
    # mpld3.enable_notebook()

    # import ma

    #FileLink
    if isinstance(instances,pd.DataFrame) or isinstance(instances,pd.Series):
        instances=instances.values
    instances=instances[:lim]
    # print(instances)
    images_per_row=elemsByRows
    images_per_row = min(len(instances), images_per_row)
    images = instances if not reshape else [np.array(instance).reshape(w,h) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((w, h * n_empty)))
    fig = plt.gcf()
    figsize = fig.get_size_inches() if figsize is None else figsize 
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    if ax is None:
        fig=plt.figure(figsize=figsize)
    if ax is None:
        plt.imshow(image, cmap = mpl.cm.binary, **options)
    else:
        ax.imshow(image, cmap = mpl.cm.binary, **options)
    if ax is None:
        plt.title(title,size=titleSize)
        plt.axis("off")
    else:
        ax.set_title(title,size=titleSize)
        ax.axis("off")
        beforeImgAx(ax)
    if not noImg:
        plt.tight_layout()
        # print(fig.get_size_inches()*fig.dpi)
        img=IMG.getImg(notDelete=True)
        plt.close()
        if show:
            # plt.show()
            # print(img.filename)
            fig=img.show(returnFig=True,show=False,figsize=figsize)
            display_html(HTML("""
<style>
g.mpld3-xaxis, g.mpld3-yaxis {
display: none;
}
</style>
            """))
            # print(img)
            # print(img.filename)
            display_html(HTML("""
                <span style='width:20px;position: absolute;' title="Save image as png">
            <a href="data:image/png;base64,{{imgData}}" download="{{filename}}"><img width="20px" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAABmJLR0QA/wD/AP+gvaeTAAAAs0lEQVRIie2WPQ6DMAxGXzJwqIrerN3pORi7cqWwtjegQymyUlMZCIlU8UleIvt7cv4BKuAG9MCQIJ7ABfBEapTkU5xkVC087mMTk4ICskqrkWOdhGntpwJ9OvNuxtgtAMU1mt81F+iRC/S9BfdScVBtrHciAM6/Epds59UqPnW7KMUdp0nee0O8RtbzY9Xk/X9rdIAOUBlQn4ETPNCKAevzYJF8Mlp4f4ca9G/X1gijd/UCDStihJWAousAAAAASUVORK5CYII="></a></span>
                """.replace("{{imgData}}",str(img.data)[2:-1]).replace("{{filename}}",filename)))
            display_html(mpld3.display())
            plt.close()
            return
        return img

def plot_ObsConfused(self,classes,preds,globalLim=10,globalNbCols=2,
        lim=10,limByPlots=100,elemsByRows=10,nbCols=2,mods=[],title=None,modelsNames=None,filename=None,titleFontsize=19,**plotConfMat_kwargs):
    from ..base import CvResultats,CrossValidItem
    if isinstance(self,CvResultats):
        return plot_ObsConfused_CvResultats(self,classes,preds,title,lim,limByPlots,elemsByRows,nbCols,**plotConfMat_kwargs)
    elif isinstance(self,CrossValidItem):
        return plot_ObsConfused_CrossValidItem(self,classes,preds,globalLim,globalNbCols,
        lim,limByPlots,elemsByRows,nbCols,mods,title,modelsNames,filename,titleFontsize,**plotConfMat_kwargs)
def plot_ObsConfused_CvResultats(self,classes,preds,title=None,lim=10,limByPlots=100,elemsByRows=10,nbCols=2,
    noImg=None,returnOK=False,**plotConfMat_kwargs):
    # from ..helpers import plotDigits
    obj=self
    from ..utils import isArr, namesEscape, T, F


    # modsN=obj.papa.papa._models.namesModels
    # models=obj.resultats

    # if len(mods)>0:
        # mods_ = [i if isStr(i) else modsN[i] for i in mods]
        # models= [obj.resultats[i] for i in mods_]
        # modelsNames_=[i for i in mods_]
        # models=dict(zip(modelsNames_,models)) if modelsNames is None else dict(zip(modelsNames,models))

    # namesY= namesEscape(namesY) if namesY is not None else namesY

    classes=classes if isArr(classes) else [classes]
    preds=preds if isArr(preds) else [preds]
    def be(fig,st,c1,p1):
        def _be(ax):
            ax.set_title('Class {} Predict {}'.format(c1,p1),fontsize= 12)
            fig.tight_layout()
            # st.set_y(1.12)

            # fig.subplots_adjust(top=0.9)
        return _be
    # print(classes)
    # print(classes)
    # print(images_per_row)
    if len(classes) >= 1:
        instances=classes
        images_per_row=nbCols
        images_per_row = min(len(instances), images_per_row)
        n_rows = (len(instances) - 1) // images_per_row + 1
        fig,ax=plt.subplots(n_rows,images_per_row, constrained_layout=False)
        st=fig.suptitle("Observations Confused for {}".format('' if obj.name is None else obj.name),
            fontsize=16)

        axes=np.ravel(ax)
        # print(preds)
        for i_,(i,j,ax) in enumerate(zip(classes,preds,axes)):
            c1=namesEscape(i)[0]
            p1=namesEscape(j)[0]
            # print(classes[0],preds[0])
            # print("la")
            dm=obj.getObsConfused(c1,p1,lim=lim)
            # ax.set_title('Class {} Predict {}'.format(c1,p1),fontsize= 12)
            resu=plotImgs(dm,lim=limByPlots,elemsByRows=elemsByRows,reshape=T,
                        title=title,noImg=(True if i_+1 < len(classes) else False) if noImg is None else noImg,beforeImgAx=be(fig,st,c1,p1) ,
                        ax=ax,**plotConfMat_kwargs)
            ax.set_title('Class {} Predict {}'.format(c1,p1),fontsize= 12)
        # for j in range()
        plt.close()
        # plt.title("Observations Confused for {}".format('' if obj.name is None else obj.name))
        return resu if returnOK else None
        # raise NotImplementedError()
    
    c1=namesEscape(classes[0])[0]
    p1=namesEscape(preds[0])[0]
    # print(classes[0],preds[0])
    dm=obj.getObsConfused(c1,p1,lim=lim)
    # if len(mods) > 0:
        # dm=dm[mods_]
    # print(dm)

    if len(dm)==0:
        print("EMPTY")
        return
    title="Class {} Predict {}".format(c1,p1) if title is None else title

    # print(title)
    return plotImgs(dm,lim=limByPlots,elemsByRows=elemsByRows,reshape=T,title=title,noImg=False,**plotConfMat_kwargs)

def plot_ObsConfused_CrossValidItem(self,classes,preds,globalLim=10,globalNbCols=2,
        lim=10,limByPlots=100,elemsByRows=10,nbCols=2,mods=[],title=None,modelsNames=None,filename=None,titleFontsize=19,**plotConfMat_kwargs):
        # from ..helpers import plotDigits
        obj=self

        modsN=obj.papa._models.namesModels
        models=obj.resultats

        if len(mods)>0:
            mods_ = [i if isStr(i) else modsN[i] for i in mods]
            models= [obj.resultats[i] for i in mods_]
            modelsNames_=[i for i in mods_]
            models=dict(zip(modelsNames_,models)) if modelsNames is None else dict(zip(modelsNames,models))

        # namesY= namesEscape(namesY) if namesY is not None else namesY
        confMatM=[plot_ObsConfused_CvResultats(v,classes,preds,lim=lim,limByPlots=limByPlots,
                                        elemsByRows=elemsByRows,returnOK=True,nbCols=nbCols,show=False,**plotConfMat_kwargs) for v in models.values()]

        title = "ObsConfused CV {}".format(obj.ID) if title is None else title
        filename="obj_confused_cv_{}.png".format(obj.ID) if filename is None else filename
        # print(confMatM)
        from ..utils import IMG_GRID
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