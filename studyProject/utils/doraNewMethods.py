from dora_study import Dora
from . import unNamesEscape, namesEscape
import collections.abc
def as_cat(self,li,escape=True,order=None):
	li = li if isinstance(li,collections.abc.Iterable) and not isinstance(li,str) else [li]
	self._data[li]=self._data[li].apply(lambda a:namesEscape(a.values) if escape else a.values,axis=0).astype("category")
	if order is not None:
		self._data[li]=self._data[li].apply(lambda a: a.cat.reorder_categories(order))
Dora.addCustomFunction(as_cat)

def names_escape(self,li,fn=int):
	li = li if isinstance(li,collections.abc.Iterable) and not isinstance(li,str) else [li]
	self._data[li]=self._data[li].apply(lambda a:namesEscape(a.values,fn),axis=0).astype("category")
Dora.addCustomFunction(names_escape)

def as_int(self,li):
	li = li if isinstance(li,collections.abc.Iterable) and not isinstance(li,str) else [li]
	self._data[li]=self._data[li].apply(lambda a:unNamesEscape(a.values),axis=0)

Dora.addCustomFunction(as_int)

def un_names_escape(self,li,fn=str):
	li = li if isinstance(li,collections.abc.Iterable) and not isinstance(li,str) else [li]
	self._data[li]=self._data[li].apply(lambda a:unNamesEscape(a.values,fn),axis=0)
Dora.addCustomFunction(un_names_escape)

