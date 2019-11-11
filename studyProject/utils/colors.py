from . import indexOfMinForValueInArray
import numpy as np
from sklearn.preprocessing import minmax_scale
RGB_BLACK='rgb(0,0,0)'
RGB_WHITE='rgb(255,255,255)'

def luminence(rgb):
    t=rgb
    luminance_ =  (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2])/255
    return luminance_

def frontColorFromColorscaleAndValue(value,colorscaleArr,colorscaleRGB,c1=RGB_BLACK,c2=RGB_WHITE):
    val=indexOfMinForValueInArray(value,colorscaleArr)
    rgb_=list(map(float,colorscaleRGB[val][len("rgb("):-1].split(",")))
    return c1 if luminence(rgb_) > 0.5 else c2

def frontColorFromColorscaleAndValues(values,colorscale,zmin=0,zmax=1,c1=RGB_BLACK,c2=RGB_WHITE):
    colorscaleArr=minmax_scale(np.array(colorscale)[:,0].astype(float),feature_range=(zmin,zmax))
    colorscaleRGB=np.array(colorscale)[:,1]
    return list(map(lambda v:frontColorFromColorscaleAndValue(v,c1=c1,c2=c2,colorscaleArr=colorscaleArr,colorscaleRGB=colorscaleRGB),values))


def flipScale(colorscale):
    colorscale=colorscale[::-1]
    colorscaleArr=np.array(colorscale)[:,0].astype(float)
    colorscaleRGB=np.array(colorscale)[:,1]
    return np.c_[(1-colorscaleArr).astype('O'),colorscaleRGB].tolist()