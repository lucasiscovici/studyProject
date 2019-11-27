from dora_study import Dora
from . import unNamesEscape, namesEscape
import collections.abc
import inspect
from functools import wraps

def has_method(o, name):
    return name in dir(o)
    
def saveLast_(self,func,*args,**kwargs):
  self._last=self._data.copy()
  self._lastlogs=self._logs.copy()

  self._lastlast=self._last.copy()
  self._lastlastlogs=self._lastlogs.copy()

  force=kwargs.pop("force",None)

  rep=func(self,*args, **kwargs)

  argss= inspect.getcallargs(func,self, *args, **kwargs)
  del argss["self"]
  argss=["{}={}".format(i,"\""+j+"\"" if isinstance(j,str) else j) for i,j in argss.items()]
  self._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) ,force=force)
  return rep

def saveLast(func):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      return saveLast_(self,func,*args,**kwargs)
  return with_logging
def as_cat(self,li,escape=True,order=None):
	if escape:
		order=namesEscape(order) if order is not None else order
	li = li if isinstance(li,collections.abc.Iterable) and not isinstance(li,str) else [li]
	self._data[li]=self._data[li].apply(lambda a:namesEscape(a.values) if escape else a.values,axis=0).astype("category")
	if order is not None:
		self._data[li]=self._data[li].apply(lambda a: a.cat.reorder_categories(order))
	return self
Dora.addCustomFunction(as_cat)

def names_escape(self,li,fn=int):
	li = li if isinstance(li,collections.abc.Iterable) and not isinstance(li,str) else [li]
	self._data[li]=self._data[li].apply(lambda a:namesEscape(a.values,fn),axis=0).astype("category")
	return self
Dora.addCustomFunction(names_escape)

def as_int(self,li):
	li = li if isinstance(li,collections.abc.Iterable) and not isinstance(li,str) else [li]
	self._data[li]=self._data[li].apply(lambda a:unNamesEscape(a.values),axis=0)
	return self
Dora.addCustomFunction(as_int)

def un_names_escape(self,li,fn=str):
	li = li if isinstance(li,collections.abc.Iterable) and not isinstance(li,str) else [li]
	self._data[li]=self._data[li].apply(lambda a:unNamesEscape(a.values,fn),axis=0)
	return self
Dora.addCustomFunction(un_names_escape)

def _executeFunc(self,fn):
    from studyPipe.studyPipe import Pipe
    import types
    if isinstance(fn,types.LambdaType):
        lambdaa=inspect.getsource(fn).strip()
        if not lambdaa.startswith("lambda"):
            raise Exception("lambda expr must alone in a line")
        fn(self._data)
    else:
        raise NotImplementedError(f"not implemented enought for '{fn.__name__}' ('{type(fn)}') ")
    
Dora.addCustomFunction(_executeFunc)
# def _executeFn(self,fn):
#     fn(self._data)
#     return self
# Dora.addCustomFunction(_executeFn)