import cufflinks as cf
import plotly_express as pe
cf.go_offline()

from interface import Interface


class Viz:
	def __init__(self,obj):
		self.obj=obj
