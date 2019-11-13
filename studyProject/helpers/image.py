import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

def plotImgs(im,title="",nr=2,nc=5,figsize=(9,5),w=28,h=28,titleSize=29,reshape=True,*args,**xargs):
    uu=reshape(im,w,h) if reshape else im
    plt.figure(figsize=figsize)
    for _i,i in enumerate(uu): 
        plt.subplot(nr,nc,_i+1)
        plt.imshow(i)
        plt.axis('off')
    plt.suptitle(title,size=titleSize);

def reshapeMultiClassif(im,w=28,h=28):
    return [reshape(j,w,h) for ww,j in im.items()]
    
def reshape(im,w=28,h=28):
    return [i.reshape(w,h) for i in im]
    
def plotDigits(instances,title="", elemsByRows=10,w=28,h=28,figsize=(9,5),
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
    if not noImg:
        beforeImgAx(ax)
        plt.tight_layout()
        # print(fig.get_size_inches()*fig.dpi)
        img=IMG.getImg(notDelete=True)
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
#             display_html(HTML("""
# <script type="text/javascript">
# function downloadURI(uri, name) 
# {
#     var link = document.createElement("a");
#     link.download = name;
#     link.href = uri;
#     link.click();
# }
# SwapPlugin=SwapPlugin
# mpld3.register_plugin("swap", SwapPlugin);
# SwapPlugin.prototype = Object.create(mpld3.Plugin.prototype);
# SwapPlugin.prototype.constructor = SwapPlugin;
# SwapPlugin.prototype.requiredProps = [];
# SwapPlugin.prototype.defaultProps =  {
#   button: true,
#   enabled: true
# };
# function SwapPlugin(fig, props) {
#   console.log("mpld3 inside SwapPlugin constructor"); 
#   mpld3.Plugin.call(this, fig, props);
# }  // End constructor here
# SwapPlugin.prototype.draw = function(){ //  Every plugin needs the prototype.draw function
#   var SwapButton = mpld3.ButtonFactory({
#     buttonID: "swap",
#     sticky: false,
#     onActivate: function() {
#         downloadURI("{{path}}","{{name}}")
#     },
#     icon: function() {
#       return("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAaCAYAAACpSkzOAAAABmJLR0QA/wD/AP+gvaeTAAAAs0lEQVRIie2WPQ6DMAxGXzJwqIrerN3pORi7cqWwtjegQymyUlMZCIlU8UleIvt7cv4BKuAG9MCQIJ7ABfBEapTkU5xkVC087mMTk4ICskqrkWOdhGntpwJ9OvNuxtgtAMU1mt81F+iRC/S9BfdScVBtrHciAM6/Epds59UqPnW7KMUdp0nee0O8RtbzY9Xk/X9rdIAOUBlQn4ETPNCKAevzYJF8Mlp4f4ca9G/X1gijd/UCDStihJWAousAAAAASUVORK5CYII=");
#     }
#   });

#   this.fig.buttons.push(SwapButton);
#   this.fig.toolbar.addButton(SwapButton); 
# };
# </script>
# """.replace("{{path}}",img.filename).replace("{{name}}",filename)))
            # return display_html(mpld3.display(fig,))
        # return ifelse(returnOK,img)