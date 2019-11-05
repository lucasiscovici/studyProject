import plotly
from ..utils import setattrAndReturnSelf
plotly.graph_objs._figure.Figure.update_config=lambda self,**kwargs: setattrAndReturnSelf(self,"_config",merge(self._config,kwargs,F))
def showshow(self,*args,**kwargs):
  meme=merge(dict(config=self._config),kwargs,F)
  return self.show(self,*args,**meme) if hasattr(self,"_config") else self.show(*args,**kwargs)
plotly.graph_objs._figure.Figure.showWithConfig=showshow
