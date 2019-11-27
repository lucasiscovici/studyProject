import pandas as pd
from collections.abc import Iterable
pd.Series.set_index = lambda self,*args,**xargs: self.set_axis(*args,**xargs,inplace=False)
pd.Series.set_name=lambda self,n:(setattr(self,"name",n),self)[1]
pd.DataFrame.dropCols = lambda self,cols,*args: self.drop(cols+list(args) if isinstance(cols,Iterable) and not isinstance(cols,str) else [cols]+list(args),axis=1)
pd.DataFrame.rename_cols = lambda self,*args,**xargs: self.set_axis(*args,**xargs,axis=1,inplace=False)
