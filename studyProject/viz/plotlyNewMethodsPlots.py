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
				titleText=False
				
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
			toImageButtonOptions=dict(filename=None,
									  width=None,
									  height=None,
									  scale=None),
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
			locales=dict(),
			**kwargs):
	"""
	staticPlot: {
	        valType: 'boolean',
	        dflt: false,
	        description: [
	            'Determines whether the graphs are interactive or not.',
	            'If *false*, no interactivity, for export or image generation.'
	        ].join(' ')
	    },

	    plotlyServerURL: {
	        valType: 'string',
	        dflt: 'https://plot.ly',
	        description: [
	            'Sets base URL for the \'Edit in Chart Studio\' (aka sendDataToCloud) mode bar button',
	            'and the showLink/sendData on-graph link'
	        ].join(' ')
	    },

	    editable: {
	        valType: 'boolean',
	        dflt: false,
	        description: [
	            'Determines whether the graph is editable or not.',
	            'Sets all pieces of `edits`',
	            'unless a separate `edits` config item overrides individual parts.'
	        ].join(' ')
	    },
	    edits: {
	        annotationPosition: {
	            valType: 'boolean',
	            dflt: false,
	            description: [
	                'Determines if the main anchor of the annotation is editable.',
	                'The main anchor corresponds to the',
	                'text (if no arrow) or the arrow (which drags the whole thing leaving',
	                'the arrow length & direction unchanged).'
	            ].join(' ')
	        },
	        annotationTail: {
	            valType: 'boolean',
	            dflt: false,
	            description: [
	                'Has only an effect for annotations with arrows.',
	                'Enables changing the length and direction of the arrow.'
	            ].join(' ')
	        },
	        annotationText: {
	            valType: 'boolean',
	            dflt: false,
	            description: 'Enables editing annotation text.'
	        },
	        axisTitleText: {
	            valType: 'boolean',
	            dflt: false,
	            description: 'Enables editing axis title text.'
	        },
	        colorbarPosition: {
	            valType: 'boolean',
	            dflt: false,
	            description: 'Enables moving colorbars.'
	        },
	        colorbarTitleText: {
	            valType: 'boolean',
	            dflt: false,
	            description: 'Enables editing colorbar title text.'
	        },
	        legendPosition: {
	            valType: 'boolean',
	            dflt: false,
	            description: 'Enables moving the legend.'
	        },
	        legendText: {
	            valType: 'boolean',
	            dflt: false,
	            description: 'Enables editing the trace name fields from the legend'
	        },
	        shapePosition: {
	            valType: 'boolean',
	            dflt: false,
	            description: 'Enables moving shapes.'
	        },
	        titleText: {
	            valType: 'boolean',
	            dflt: false,
	            description: 'Enables editing the global layout title.'
	        }
	    },

	    autosizable: {
	        valType: 'boolean',
	        dflt: false,
	        description: [
	            'Determines whether the graphs are plotted with respect to',
	            'layout.autosize:true and infer its container size.'
	        ].join(' ')
	    },
	    responsive: {
	        valType: 'boolean',
	        dflt: false,
	        description: [
	            'Determines whether to change the layout size when window is resized.',
	            'In v2, this option will be removed and will always be true.'
	        ].join(' ')
	    },
	    fillFrame: {
	        valType: 'boolean',
	        dflt: false,
	        description: [
	            'When `layout.autosize` is turned on, determines whether the graph',
	            'fills the container (the default) or the screen (if set to *true*).'
	        ].join(' ')
	    },
	    frameMargins: {
	        valType: 'number',
	        dflt: 0,
	        min: 0,
	        max: 0.5,
	        description: [
	            'When `layout.autosize` is turned on, set the frame margins',
	            'in fraction of the graph size.'
	        ].join(' ')
	    },

	    scrollZoom: {
	        valType: 'flaglist',
	        flags: ['cartesian', 'gl3d', 'geo', 'mapbox'],
	        extras: [true, false],
	        dflt: 'gl3d+geo+mapbox',
	        description: [
	            'Determines whether mouse wheel or two-finger scroll zooms is enable.',
	            'Turned on by default for gl3d, geo and mapbox subplots',
	            '(as these subplot types do not have zoombox via pan),',
	            'but turned off by default for cartesian subplots.',
	            'Set `scrollZoom` to *false* to disable scrolling for all subplots.'
	        ].join(' ')
	    },
	    doubleClick: {
	        valType: 'enumerated',
	        values: [false, 'reset', 'autosize', 'reset+autosize'],
	        dflt: 'reset+autosize',
	        description: [
	            'Sets the double click interaction mode.',
	            'Has an effect only in cartesian plots.',
	            'If *false*, double click is disable.',
	            'If *reset*, double click resets the axis ranges to their initial values.',
	            'If *autosize*, double click set the axis ranges to their autorange values.',
	            'If *reset+autosize*, the odd double clicks resets the axis ranges',
	            'to their initial values and even double clicks set the axis ranges',
	            'to their autorange values.'
	        ].join(' ')
	    },
	    doubleClickDelay: {
	        valType: 'number',
	        dflt: 300,
	        min: 0,
	        description: [
	            'Sets the delay for registering a double-click in ms.',
	            'This is the time interval (in ms) between first mousedown and',
	            '2nd mouseup to constitute a double-click.',
	            'This setting propagates to all on-subplot double clicks',
	            '(except for geo and mapbox) and on-legend double clicks.'
	        ].join(' ')
	    },

	    showAxisDragHandles: {
	        valType: 'boolean',
	        dflt: true,
	        description: [
	            'Set to *false* to omit cartesian axis pan/zoom drag handles.'
	        ].join(' ')
	    },
	    showAxisRangeEntryBoxes: {
	        valType: 'boolean',
	        dflt: true,
	        description: [
	            'Set to *false* to omit direct range entry at the pan/zoom drag points,',
	            'note that `showAxisDragHandles` must be enabled to have an effect.'
	        ].join(' ')
	    },

	    showTips: {
	        valType: 'boolean',
	        dflt: true,
	        description: [
	            'Determines whether or not tips are shown while interacting',
	            'with the resulting graphs.'
	        ].join(' ')
	    },

	    showLink: {
	        valType: 'boolean',
	        dflt: false,
	        description: [
	            'Determines whether a link to plot.ly is displayed',
	            'at the bottom right corner of resulting graphs.',
	            'Use with `sendData` and `linkText`.'
	        ].join(' ')
	    },
	    linkText: {
	        valType: 'string',
	        dflt: 'Edit chart',
	        noBlank: true,
	        description: [
	            'Sets the text appearing in the `showLink` link.'
	        ].join(' ')
	    },
	    sendData: {
	        valType: 'boolean',
	        dflt: true,
	        description: [
	            'If *showLink* is true, does it contain data',
	            'just link to a plot.ly file?'
	        ].join(' ')
	    },
	    showSources: {
	        valType: 'any',
	        dflt: false,
	        description: [
	            'Adds a source-displaying function to show sources on',
	            'the resulting graphs.'
	        ].join(' ')
	    },

	    displayModeBar: {
	        valType: 'enumerated',
	        values: ['hover', true, false],
	        dflt: 'hover',
	        description: [
	            'Determines the mode bar display mode.',
	            'If *true*, the mode bar is always visible.',
	            'If *false*, the mode bar is always hidden.',
	            'If *hover*, the mode bar is visible while the mouse cursor',
	            'is on the graph container.'
	        ].join(' ')
	    },
	    showSendToCloud: {
	        valType: 'boolean',
	        dflt: false,
	        description: [
	            'Should we include a ModeBar button, labeled "Edit in Chart Studio",',
	            'that sends this chart to plot.ly or another plotly server as specified',
	            'by `plotlyServerURL` for editing, export, etc? Prior to version 1.43.0',
	            'this button was included by default, now it is opt-in using this flag.',
	            'Note that this button can (depending on `plotlyServerURL`) send your data',
	            'to an external server. However that server does not persist your data',
	            'until you arrive at the Chart Studio and explicitly click "Save".'
	        ].join(' ')
	    },
	    showEditInChartStudio: {
	        valType: 'boolean',
	        dflt: false,
	        description: [
	            'Same as `showSendToCloud`, but use a pencil icon instead of a floppy-disk.',
	            'Note that if both `showSendToCloud` and `showEditInChartStudio` are turned,',
	            'only `showEditInChartStudio` will be honored.'
	        ].join(' ')
	    },
	    modeBarButtonsToRemove: {
	        valType: 'any',
	        values:["toImage", 
					"sedDataToCloud", 
					"editIChartStudio", 
					"zoom2d", 
					"pa2d", 
					"select2d", 
					"lasso2d", 
					"zoomI2d", 
					"zoomOut2d", 
					"autoScale2d", 
					"resetScale2d", 
					"hoverClosestCartesia", 
					"hoverCompareCartesia", 
					"zoom3d", 
					"pa3d", 
					"orbitRotatio", 
					"tableRotatio", 
					"resetCameraDefault3d", 
					"resetCameraLastSave3d", 
					"hoverClosest3d", 
					"zoomIGeo", 
					"zoomOutGeo", 
					"resetGeo", 
					"hoverClosestGeo", 
					"hoverClosestGl2d", 
					"hoverClosestPie", 
					"resetViewSakey", 
					"toggleHover", 
					"resetViews", 
					"toggleSpikelies", 
					"resetViewMapbox"],
	        dflt: [],
	        description: [
	            'Remove mode bar buttons by name.',
	            'See ./components/modebar/buttons.js for the list of names.'
	        ].join(' ')
	    },
	    modeBarButtonsToAdd: {
	        valType: 'any',
	        dflt: [],
	        description: [
	            'Add mode bar button using config objects',
	            'See ./components/modebar/buttons.js for list of arguments.'
	        ].join(' ')
	    },
	    modeBarButtons: {
	        valType: 'any',
	        dflt: false,
	        description: [
	            'Define fully custom mode bar buttons as nested array,',
	            'where the outer arrays represents button groups, and',
	            'the inner arrays have buttons config objects or names of default buttons',
	            'See ./components/modebar/buttons.js for more info.'
	        ].join(' ')
	    },
	    toImageButtonOptions: {
	        valType: 'any',
	        dflt: {},
	        description: [
	            'Statically override options for toImage modebar button',
	            'allowed keys are format, filename, width, height, scale',
	            'see ../components/modebar/buttons.js'
	        ].join(' ')
	    },
	    displaylogo: {
	        valType: 'boolean',
	        dflt: true,
	        description: [
	            'Determines whether or not the plotly logo is displayed',
	            'on the end of the mode bar.'
	        ].join(' ')
	    },
	    watermark: {
	        valType: 'boolean',
	        dflt: false,
	        description: 'watermark the images with the company\'s logo'
	    },

	    plotGlPixelRatio: {
	        valType: 'number',
	        dflt: 2,
	        min: 1,
	        max: 4,
	        description: [
	            'Set the pixel ratio during WebGL image export.',
	            'This config option was formerly named `plot3dPixelRatio`',
	            'which is now deprecated.'
	        ].join(' ')
	    },

	    setBackground: {
	        valType: 'any',
	        dflt: 'transparent',
	        description: [
	            'Set function to add the background color (i.e. `layout.paper_color`)',
	            'to a different container.',
	            'This function take the graph div as first argument and the current background',
	            'color as second argument.',
	            'Alternatively, set to string *opaque* to ensure there is white behind it.'
	        ].join(' ')
	    },

	    topojsonURL: {
	        valType: 'string',
	        noBlank: true,
	        dflt: 'https://cdn.plot.ly/',
	        description: [
	            'Set the URL to topojson used in geo charts.',
	            'By default, the topojson files are fetched from cdn.plot.ly.',
	            'For example, set this option to:',
	            '<path-to-plotly.js>/dist/topojson/',
	            'to render geographical feature using the topojson files',
	            'that ship with the plotly.js module.'
	        ].join(' ')
	    },

	    mapboxAccessToken: {
	        valType: 'string',
	        dflt: null,
	        description: [
	            'Mapbox access token (required to plot mapbox trace types)',
	            'If using an Mapbox Atlas server, set this option to \'\'',
	            'so that plotly.js won\'t attempt to authenticate to the public Mapbox server.'
	        ].join(' ')
	    },

	    logging: {
	        valType: 'boolean',
	        dflt: 1,
	        description: [
	            'Turn all console logging on or off (errors will be thrown)',
	            'This should ONLY be set via Plotly.setPlotConfig',
	            'Available levels:',
	            '0: no logs',
	            '1: warnings and errors, but not informational messages',
	            '2: verbose logs'
	        ].join(' ')
	    },

	    queueLength: {
	        valType: 'integer',
	        min: 0,
	        dflt: 0,
	        description: 'Sets the length of the undo/redo queue.'
	    },

	    globalTransforms: {
	        valType: 'any',
	        dflt: [],
	        description: [
	            'Set global transform to be applied to all traces with no',
	            'specification needed'
	        ].join(' ')
	    },

	    locale: {
	        valType: 'string',
	        dflt: 'en-US',
	        description: [
	            'Which localization should we use?',
	            'Should be a string like \'en\' or \'en-US\'.'
	        ].join(' ')
	    },

	    locales: {
	        valType: 'any',
	        dflt: {},
	        description: [
	            'Localization definitions',
	            'Locales can be provided either here (specific to one chart) or globally',
	            'by registering them as modules.',
	            'Should be an object of objects {locale: {dictionary: {...}, format: {...}}}',
	            '{',
	            '  da: {',
	            '      dictionary: {\'Reset axes\': \'Nulstil aksler\', ...},',
	            '      format: {months: [...], shortMonths: [...]}',
	            '  },',
	            '  ...',
	            '}',
	            'All parts are optional. When looking for translation or format fields, we',
	            'look first for an exact match in a config locale, then in a registered',
	            'module. If those fail, we strip off any regionalization (\'en-US\' -> \'en\')',
	            'and try each (config, registry) again. The final fallback for translation',
	            'is untranslated (which is US English) and for formats is the base English',
	            '(the only consequence being the last fallback date format %x is DD/MM/YYYY',
	            'instead of MM/DD/YYYY). Currently `grouping` and `currency` are ignored',
	            'for our automatic number formatting, but can be used in custom formats.'
	        ].join(' ')
	    }
	};

	"""
	return setattrAndReturnSelf(self,"_config",merge(self._config,dict(staticPlot=staticPlot,
plotlyServerURL=plotlyServerURL,
editable=editable,
edits=edits,
autosizable=autosizable,
responsive=responsive,
fillFrame=fillFrame,
frameMargins=frameMargins,
scrollZoom=scrollZoom,
doubleClick=doubleClick,
# doubleClickDelay=doubleClickDelay,
showAxisDragHandles=showAxisDragHandles,
showAxisRangeEntryBoxes=showAxisRangeEntryBoxes,
showTips=showTips,
showLink=showLink,
linkText=linkText,
sendData=sendData,
showSources=showSources,
displayModeBar=displayModeBar,
showSendToCloud=showSendToCloud,
# showEditInChartStudio=showEditInChartStudio,
modeBarButtonsToRemove=modeBarButtonsToRemove,
modeBarButtonsToAdd=modeBarButtonsToAdd,
modeBarButtons=modeBarButtons,
toImageButtonOptions=toImageButtonOptions,
displaylogo=displaylogo,
watermark=watermark,
plotGlPixelRatio=plotGlPixelRatio,
setBackground=setBackground,
topojsonURL=topojsonURL,
mapboxAccessToken=mapboxAccessToken,
logging=logging,
queueLength=queueLength,
globalTransforms=globalTransforms,
locale=locale,
locales=locales),defaults=dict(
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
				titleText=False
				
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
			toImageButtonOptions=dict(filename=None,
									  width=None,
									  height=None,
									  scale=None),
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
),add=F))

plotly.graph_objs._figure.Figure.update_config=upConf
def showshow(self,*args,**kwargs):
	kwargs= {} if kwargs is None else kwargs
	meme=merge(dict(config=self._config),kwargs,add=F)
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