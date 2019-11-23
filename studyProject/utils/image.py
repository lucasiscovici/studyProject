import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from . import TMP_FILE
import numpy as np
import base64
from . import ifOneGetArr
import operator
from functools import reduce
class IMG:
    def __init__(self,im):
        self.im=im
        self.filename=None
        self.figsize=None


    @property
    def data(self):
        return  base64.b64encode(open(self.filename, "rb").read())
    
    def show(self,figure=None,returnFig=False,show=True,figsize=None,dpi=None,cmap=plt.cm.gray_r):
        img=self.im
        #if not figure:
        #    plt.figure(figsize=figsize,dpi=dpi)
        fig = plt.gcf()
        size = fig.get_size_inches() if figsize is None else figsize 
        dpi= fig.dpi if dpi is None else dpi
        fig, ax = plt.subplots(figsize=figsize,dpi=dpi)
        ax.axis("off")
        oe=ax.imshow(img,cmap=cmap)
        oe.axes.get_xaxis().set_visible(False)
        oe.axes.get_yaxis().set_visible(False)
        fig.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        if show:
            plt.show()
        if returnFig:
            return fig
    
    @staticmethod
    def getImg(name=None,ext="png",dpi=400,notDelete=False,**xargs):
        tmpF=TMP_FILE()
        filename=name if name is not None else  tmpF.get_filename(ext=ext)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            #hspace = 0, wspace = 0)
        # plt.show()
        fs=plt.gcf().get_size_inches()
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight',pad_inches = 0,format=ext, dpi=dpi,**xargs)
        plt.close()
        im=IMG(mpimg.imread(filename))
        im.figsize=fs
        if not notDelete:
            tmpF.delete()
        else:
            im.filename=filename
        #print(filename)
        return im
    @staticmethod
    def fromPath(p):
        return IMG(mpimg.imread(p))

class IMG_GRID:
    @staticmethod
    def grid(imgs,nbCols=5,cmap=plt.cm.gray_r,toImg=False,title=None,titleFontsize=15,**xargs):
        instances=imgs
        elemsByRows=nbCols
        #instances=instances[:lim]
        images_per_row=elemsByRows
        images_per_row = min(len(instances), images_per_row)
        n_rows = (len(instances) - 1) // images_per_row + 1
        #row_images = []
        #n_empty = n_rows * images_per_row - len(instances)
        figsize=reduce(operator.add,[np.array(list(i.figsize)) for i in imgs])
        # print(figsize)
        f, axs2 = plt.subplots(n_rows,images_per_row,figsize=figsize,**xargs)
        if title is not None:
            f.suptitle(title,fontsize=titleFontsize)
        axs = np.array(ifOneGetArr(axs2,images_per_row*n_rows)).flatten()
        d=0
        for img, ax in zip(instances, axs):
            #rm=ax.imshow(img.im)
            oe=ax.imshow(img.im,cmap=cmap)
            oe.axes.get_xaxis().set_visible(False)
            oe.axes.get_yaxis().set_visible(False)
            ax.axis('off')
            d+=1
        if d< len(axs):
            #print(axs[d:])
            for ax in axs[d:]:
                #print("la")
                ax.set_axis_off()
                ax.set_visible(False)
        #.set_axis_off()
        #plt.gca().set_aspect('equal', adjustable='datalim')
        # plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #plt.subplots_adjust(bottom=0,right=0)
        # plt.show()
        if toImg:
            return IMG.getImg(notDelete=True)
        else:
            plt.show()
                