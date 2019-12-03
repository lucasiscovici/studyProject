import matplotlib.pyplot as plt
addButonSave="""
<script>
/*
* FileSaver.js
* A saveAs() FileSaver implementation.
*
* By Eli Grey, http://eligrey.com
*
* License : https://github.com/eligrey/FileSaver.js/blob/master/LICENSE.md (MIT)
* source  : http://purl.eligrey.com/github/FileSaver.js
*/

// The one and only way of getting global scope in all environments
// https://stackoverflow.com/q/3277182/1008999
var _global = typeof window === 'object' && window.window === window
  ? window : typeof self === 'object' && self.self === self
  ? self : typeof global === 'object' && global.global === global
  ? global
  : this

function bom (blob, opts) {
  if (typeof opts === 'undefined') opts = { autoBom: false }
  else if (typeof opts !== 'object') {
    console.warn('Deprecated: Expected third argument to be a object')
    opts = { autoBom: !opts }
  }

  // prepend BOM for UTF-8 XML and text/* types (including HTML)
  // note: your browser will automatically convert UTF-16 U+FEFF to EF BB BF
  if (opts.autoBom && /^\s*(?:text\/\S*|application\/xml|\S*\/\S*\+xml)\s*;.*charset\s*=\s*utf-8/i.test(blob.type)) {
    return new Blob([String.fromCharCode(0xFEFF), blob], { type: blob.type })
  }
  return blob
}

function download (url, name, opts) {
  var xhr = new XMLHttpRequest()
  xhr.open('GET', url)
  xhr.responseType = 'blob'
  xhr.onload = function () {
    saveAs(xhr.response, name, opts)
  }
  xhr.onerror = function () {
    console.error('could not download file')
  }
  xhr.send()
}

function corsEnabled (url) {
  var xhr = new XMLHttpRequest()
  // use sync to avoid popup blocker
  xhr.open('HEAD', url, false)
  try {
    xhr.send()
  } catch (e) {}
  return xhr.status >= 200 && xhr.status <= 299
}

// `a.click()` doesn't work for all browsers (#465)
function click (node) {
  try {
    node.dispatchEvent(new MouseEvent('click'))
  } catch (e) {
    var evt = document.createEvent('MouseEvents')
    evt.initMouseEvent('click', true, true, window, 0, 0, 0, 80,
                          20, false, false, false, false, 0, null)
    node.dispatchEvent(evt)
  }
}

var saveAs = _global.saveAs || (
  // probably in some web worker
  (typeof window !== 'object' || window !== _global)
    ? function saveAs () { /* noop */ }

  // Use download attribute first if possible (#193 Lumia mobile)
  : 'download' in HTMLAnchorElement.prototype
  ? function saveAs (blob, name, opts) {
    var URL = _global.URL || _global.webkitURL
    var a = document.createElement('a')
    name = name || blob.name || 'download'

    a.download = name
    a.rel = 'noopener' // tabnabbing

    // TODO: detect chrome extensions & packaged apps
    // a.target = '_blank'

    if (typeof blob === 'string') {
      // Support regular links
      a.href = blob
      if (a.origin !== location.origin) {
        corsEnabled(a.href)
          ? download(blob, name, opts)
          : click(a, a.target = '_blank')
      } else {
        click(a)
      }
    } else {
      // Support blobs
      a.href = URL.createObjectURL(blob)
      setTimeout(function () { URL.revokeObjectURL(a.href) }, 4E4) // 40s
      setTimeout(function () { click(a) }, 0)
    }
  }

  // Use msSaveOrOpenBlob as a second approach
  : 'msSaveOrOpenBlob' in navigator
  ? function saveAs (blob, name, opts) {
    name = name || blob.name || 'download'

    if (typeof blob === 'string') {
      if (corsEnabled(blob)) {
        download(blob, name, opts)
      } else {
        var a = document.createElement('a')
        a.href = blob
        a.target = '_blank'
        setTimeout(function () { click(a) })
      }
    } else {
      navigator.msSaveOrOpenBlob(bom(blob, opts), name)
    }
  }

  // Fallback to using FileReader and a popup
  : function saveAs (blob, name, opts, popup) {
    // Open a popup immediately do go around popup blocker
    // Mostly only available on user interaction and the fileReader is async so...
    popup = popup || open('', '_blank')
    if (popup) {
      popup.document.title =
      popup.document.body.innerText = 'downloading...'
    }

    if (typeof blob === 'string') return download(blob, name, opts)

    var force = blob.type === 'application/octet-stream'
    var isSafari = /constructor/i.test(_global.HTMLElement) || _global.safari
    var isChromeIOS = /CriOS\/[\d]+/.test(navigator.userAgent)

    if ((isChromeIOS || (force && isSafari)) && typeof FileReader !== 'undefined') {
      // Safari doesn't allow downloading of blob URLs
      var reader = new FileReader()
      reader.onloadend = function () {
        var url = reader.result
        url = isChromeIOS ? url : url.replace(/^data:[^;]*;/, 'data:attachment/file;')
        if (popup) popup.location.href = url
        else location = url
        popup = null // reverse-tabnabbing #460
      }
      reader.readAsDataURL(blob)
    } else {
      var URL = _global.URL || _global.webkitURL
      var url = URL.createObjectURL(blob)
      if (popup) popup.location = url
      else location.href = url
      popup = null // reverse-tabnabbing #460
      setTimeout(function () { URL.revokeObjectURL(url) }, 4E4) // 40s
    }
  }
)

_global.saveAs = saveAs.saveAs = saveAs

if (typeof module !== 'undefined') {
  module.exports = saveAs;
}
</script>
<script>
/*
 * JavaScript Canvas to Blob
 * https://github.com/blueimp/JavaScript-Canvas-to-Blob
 *
 * Copyright 2012, Sebastian Tschan
 * https://blueimp.net
 *
 * Licensed under the MIT license:
 * https://opensource.org/licenses/MIT
 *
 * Based on stackoverflow user Stoive's code snippet:
 * http://stackoverflow.com/q/4998908
 */

/* global define, Uint8Array, ArrayBuffer, module */

;(function(window) {
  'use strict'

  var CanvasPrototype =
    window.HTMLCanvasElement && window.HTMLCanvasElement.prototype
  var hasBlobConstructor =
    window.Blob &&
    (function() {
      try {
        return Boolean(new Blob())
      } catch (e) {
        return false
      }
    })()
  var hasArrayBufferViewSupport =
    hasBlobConstructor &&
    window.Uint8Array &&
    (function() {
      try {
        return new Blob([new Uint8Array(100)]).size === 100
      } catch (e) {
        return false
      }
    })()
  var BlobBuilder =
    window.BlobBuilder ||
    window.WebKitBlobBuilder ||
    window.MozBlobBuilder ||
    window.MSBlobBuilder
  var dataURIPattern = /^data:((.*?)(;charset=.*?)?)(;base64)?,/
  var dataURLtoBlob =
    (hasBlobConstructor || BlobBuilder) &&
    window.atob &&
    window.ArrayBuffer &&
    window.Uint8Array &&
    function(dataURI) {
      var matches,
        mediaType,
        isBase64,
        dataString,
        byteString,
        arrayBuffer,
        intArray,
        i,
        bb
      // Parse the dataURI components as per RFC 2397
      matches = dataURI.match(dataURIPattern)
      if (!matches) {
        throw new Error('invalid data URI')
      }
      // Default to text/plain;charset=US-ASCII
      mediaType = matches[2]
        ? matches[1]
        : 'text/plain' + (matches[3] || ';charset=US-ASCII')
      isBase64 = !!matches[4]
      dataString = dataURI.slice(matches[0].length)
      if (isBase64) {
        // Convert base64 to raw binary data held in a string:
        byteString = atob(dataString)
      } else {
        // Convert base64/URLEncoded data component to raw binary:
        byteString = decodeURIComponent(dataString)
      }
      // Write the bytes of the string to an ArrayBuffer:
      arrayBuffer = new ArrayBuffer(byteString.length)
      intArray = new Uint8Array(arrayBuffer)
      for (i = 0; i < byteString.length; i += 1) {
        intArray[i] = byteString.charCodeAt(i)
      }
      // Write the ArrayBuffer (or ArrayBufferView) to a blob:
      if (hasBlobConstructor) {
        return new Blob([hasArrayBufferViewSupport ? intArray : arrayBuffer], {
          type: mediaType
        })
      }
      bb = new BlobBuilder()
      bb.append(arrayBuffer)
      return bb.getBlob(mediaType)
    }
  if (window.HTMLCanvasElement && !CanvasPrototype.toBlob) {
    if (CanvasPrototype.mozGetAsFile) {
      CanvasPrototype.toBlob = function(callback, type, quality) {
        var self = this
        setTimeout(function() {
          if (quality && CanvasPrototype.toDataURL && dataURLtoBlob) {
            callback(dataURLtoBlob(self.toDataURL(type, quality)))
          } else {
            callback(self.mozGetAsFile('blob', type))
          }
        })
      }
    } else if (CanvasPrototype.toDataURL && dataURLtoBlob) {
      CanvasPrototype.toBlob = function(callback, type, quality) {
        var self = this
        setTimeout(function() {
          callback(dataURLtoBlob(self.toDataURL(type, quality)))
        })
      }
    }
  }
  if (typeof define === 'function' && define.amd) {
    define(function() {
      return dataURLtoBlob
    })
  } else if (typeof module === 'object' && module.exports) {
    module.exports = dataURLtoBlob
  } else {
    window.dataURLtoBlob = dataURLtoBlob
  }
})(window)
</script>
<script>
// Set-up the export button
d3.select('#saveButton').on('click', function(){
    svg=$(this).parent().parent().parent().find(".mpld3-figure")[0]
	var svgString = getSVGString(svg);
    width=$(svg).attr("width")
    height=$(svg).attr("height")
	svgString2Image( svgString, 2*width, 2*height, 'png', save ); // passes Blob and filesize String to the callback

	function save( dataBlob, filesize ){
		saveAs( dataBlob, 'D3 vis exported to PNG.png' ); // FileSaver.js function
	}
});

// Below are the functions that handle actual exporting:
// getSVGString ( svgNode ) and svgString2Image( svgString, width, height, format, callback )
function getSVGString( svgNode ) {
	svgNode.setAttribute('xlink', 'http://www.w3.org/1999/xlink');
	var cssStyleText = getCSSStyles( svgNode );
	appendCSS( cssStyleText, svgNode );

	var serializer = new XMLSerializer();
	var svgString = serializer.serializeToString(svgNode);
	svgString = svgString.replace(/(\w+)?:?xlink=/g, 'xmlns:xlink='); // Fix root xlink without namespace
	svgString = svgString.replace(/NS\d+:href/g, 'xlink:href'); // Safari NS namespace fix

	return svgString;

	function getCSSStyles( parentElement ) {
		var selectorTextArr = [];

		// Add Parent element Id and Classes to the list
		selectorTextArr.push( '#'+parentElement.id );
		for (var c = 0; c < parentElement.classList.length; c++)
				if ( !contains('.'+parentElement.classList[c], selectorTextArr) )
					selectorTextArr.push( '.'+parentElement.classList[c] );

		// Add Children element Ids and Classes to the list
		var nodes = parentElement.getElementsByTagName("*");
		for (var i = 0; i < nodes.length; i++) {
			var id = nodes[i].id;
			if ( !contains('#'+id, selectorTextArr) )
				selectorTextArr.push( '#'+id );

			var classes = nodes[i].classList;
			for (var c = 0; c < classes.length; c++)
				if ( !contains('.'+classes[c], selectorTextArr) )
					selectorTextArr.push( '.'+classes[c] );
		}

		// Extract CSS Rules
		var extractedCSSText = "";
		for (var i = 0; i < document.styleSheets.length; i++) {
			var s = document.styleSheets[i];
			
			try {
			    if(!s.cssRules) continue;
			} catch( e ) {
		    		if(e.name !== 'SecurityError') throw e; // for Firefox
		    		continue;
		    	}

			var cssRules = s.cssRules;
			for (var r = 0; r < cssRules.length; r++) {
				if ( contains( cssRules[r].selectorText, selectorTextArr ) )
					extractedCSSText += cssRules[r].cssText;
			}
		}
		

		return extractedCSSText;

		function contains(str,arr) {
			return arr.indexOf( str ) === -1 ? false : true;
		}

	}

	function appendCSS( cssText, element ) {
		var styleElement = document.createElement("style");
		styleElement.setAttribute("type","text/css"); 
		styleElement.innerHTML = cssText;
		var refNode = element.hasChildNodes() ? element.children[0] : null;
		element.insertBefore( styleElement, refNode );
	}
}


function svgString2Image( svgString, width, height, format, callback ) {
	var format = format ? format : 'png';

	var imgsrc = 'data:image/svg+xml;base64,'+ btoa( unescape( encodeURIComponent( svgString ) ) ); // Convert SVG string to data URL

	var canvas = document.createElement("canvas");
	var context = canvas.getContext("2d");

	canvas.width = width;
	canvas.height = height;

	var image = new Image();
	image.onload = function() {
		context.clearRect ( 0, 0, width, height );
		context.drawImage(image, 0, 0, width, height);

		canvas.toBlob( function(blob) {
			var filesize = Math.round( blob.length/1024 ) + ' KB';
			if ( callback ) callback( blob, filesize );
		});

		
	};

	image.src = imgsrc;
}
</script>
"""
def addSaveButtonMpld3():
    from IPython.display import display, HTML
    display(HTML("""
    <button id='saveButton'><img width="20px" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAABmJLR0QA/wD/AP+gvaeTAAAAs0lEQVRIie2WPQ6DMAxGXzJwqIrerN3pORi7cqWwtjegQymyUlMZCIlU8UleIvt7cv4BKuAG9MCQIJ7ABfBEapTkU5xkVC087mMTk4ICskqrkWOdhGntpwJ9OvNuxtgtAMU1mt81F+iRC/S9BfdScVBtrHciAM6/Epds59UqPnW7KMUdp0nee0O8RtbzY9Xk/X9rdIAOUBlQn4ETPNCKAevzYJF8Mlp4f4ca9G/X1gijd/UCDStihJWAousAAAAASUVORK5CYII="></button>
            """+addButonSave))
import mpld3_study
from IPython.display import display, HTML
from mpld3_study import display
import numpy as np
def add_hover_label_to_rect(ax=None,roundVal=2):
    if ax is None:
      ax=plt.gca()
    if not isinstance(ax,list):
      ax=[ax]
    m=[get_datas_plt(i) for i in ax]
    #print(m)
    dd=[]
    for j in m:
        for bars in j:
            for bar in bars:
                tooltip = mpld3_study.plugins.LineLabelTooltip(bar, label=str(np.round(bar.get_height(),roundVal)))
                dd.append(tooltip)
    return dd

def add_select_filter(ax):
    m=[get_datas_plt(i) for i in ax]
    k=[]
    for j,mi in zip(ax,m):
        handles, labels2=j.get_legend_handles_labels()
        interactive_legend = mpld3_study.plugins.InteractiveLegendPlugin(mi,
                                                             labels2,
                                                             alpha_unsel=0,ax=j,
                                                             alpha_over=1.5, 
                                                             start_visible=True)
        k+=[interactive_legend]
    return k
def get_datas_plt(ax):
    ii=[]
    o=[]
    for i, bar in enumerate(ax.get_children()):
        if bar.__class__.__name__=="Rectangle" and bar.get_label() != "":
            if bar.get_label()!="_nolegend_":
                if len(o)>0:
                    ii.append(o)
                o=[]
            o.append(bar)
    if len(o)>0:
        ii.append(o)
        
    return ii
def connect_mpld3(plugins,fig=None,addDefaults=True):
    fig=fig if fig is not None else plt.gcf()
    if addDefaults:
        plugins=mpld3_study.plugins.get_plugins(fig)+plugins
    mpld3_study.plugins.clear(fig)
    mpld3_study.plugins.connect(fig,*plugins)