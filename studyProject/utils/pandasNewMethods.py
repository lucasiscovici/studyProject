import pandas as pd

pd.Series.set_index = lambda self,*args,**xargs: self.set_axis(*args,**xargs,inplace=False)
pd.Series.set_name=lambda self,n:(setattr(self,"name",n),self)[1]