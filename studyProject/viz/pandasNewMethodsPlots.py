import plotly_study.figure_factory as ff
from plotly_study import offline
from ..utils import F, merge
import pandas as pd
def table_plot(self,filename="table_plot",
					returnFig=True,
					plotFig=False,
					create_table_xargs=dict(),
					iplot_xargs=dict()):
	_create_table_xargs=dict(index=True)
	create_table_xargs=merge(_create_table_xargs,create_table_xargs,add=F)
	table = ff.create_table(self,**create_table_xargs)
	table.update_config(filename='table_plot')
	if plotFig:
		fig=table.show()
		# offline.iplot(table, filename='table_plot',**iplot_xargs)
	return (fig if plotFig else table) if returnFig else None

pd.DataFrame.table_plot = table_plot