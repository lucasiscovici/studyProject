import plotly
from ..utils import F, merge
from ..utils import setattrAndReturnSelf


def upConf(self,
			staticPlot=False,
plotlyServerURL="https://plot.ly",
editable=False,
edits=dict(
	annotationPosition=False,
	annotationTail=False,
	annotationText=False,
	axisTitleText=False,
	colorbarPosition=False,
	colorbarTitleText=False,
	legendPosition=False,
	legendText=False,
	shapePosition=False,
	titleText=False,
	
),
autosizable=False,
responsive=False,
fillFrame=False,
frameMargins=0,
scrollZoom="gl3d+geo+mapbox",
doubleClick="reset+autosize",
doubleClickDelay=300,
showAxisDragHandles=True,
showAxisRangeEntryBoxes=True,
showTips=True,
showLink=False,
linkText="Edit chart",
sendData=True,
showSources=False,
displayModeBar="hover",
showSendToCloud=False,
showEditInChartStudio=False,
modeBarButtonsToRemove=[],
modeBarButtonsToAdd=[],
modeBarButtons=False,
toImageButtonOptions=dict(),
displaylogo=True,
watermark=False,
plotGlPixelRatio=2,
setBackground="transparent",
topojsonURL="https://cdn.plot.ly/",
mapboxAccessToken="https://cdn.plot.ly/",
logging=1,
queueLength=0,
globalTransforms=[],
locale="en-US",
locales=dict()
			**kwargs):
	return setattrAndReturnSelf(self,"_config",merge(self._config,kwargs,F))

plotly.graph_objs._figure.Figure.update_config=upConf
def showshow(self,*args,**kwargs):
  meme=merge(dict(config=self._config),kwargs,F)
  return self.show(*args,**meme) if hasattr(self,"_config") else self.show(*args,**kwargs)
plotly.graph_objs._figure.Figure.showWithConfig=showshow


# lasFrepre=plotly.graph_objs._figure.Figure.__repr__
# plotly.graph_objs._figure.Figure.__repr2__=lasFrepre
# plotly.graph_objs._figure.Figure.__repr__=lambda self: self.showWithConfig()


def ipython_display2(self):
    import plotly.io as pio

    if pio.renderers.render_on_display and pio.renderers.default:
        if hasattr(self,"showWithConfig"):
        	self.showWithConfig()
        else:
        	pio.show(self)
    else:
        print (repr(self))

figIdisplay=plotly.graph_objs._figure.Figure._ipython_display_
plotly.graph_objs._figure.Figure._ipython_display_=ipython_display2
plotly.graph_objs._figure.Figure._ipython_display2_=figIdisplay

# function ttt(l, tab=""){
# 	var srti=tab
# 	var val = null;
#     for (const [fruit, count] of  Object.entries(l)) {
#     	if(Array.isArray(count)){
#     		val="[]"
#     	}else if(count == true && typeof(count) === "boolean"){
#     		val="True"
#     	}else if(count==false && typeof(count) === "boolean"){
#     		val="False"
#     	}else if(typeof(count)=="string" || typeof(count)=="number"){
#     	val = count
#     	}
#     	else if (typeof(count) === 'object' && count !== null){
#     		if (Object.keys(count).length==0){
#     			val="dict()"
#     		}else{
#     			val="dict(\n"+ttt(count,tab+"\t")+"\n)"
#     		}
#     	}
#     	//console.log(fruit+" "+typeof(count)+" "+val)
# 		srti=srti+fruit+"="+val+"\n"+tab
#     }
#     return srti 
# }    