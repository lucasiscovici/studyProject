from speedml_study import Speedml
from studyPipe import df, X
import numpy as np
import pandas as pd
from . import get_args, StudyClass
import inspect
from functools import wraps
import warnings
import copy
def has_method(o, name):
    return name in dir(o)

def copy(ld):
  if ld is None:
    return ld
  return ld.copy()

def saveLast_(self,func,*args,**kwargs):
  self._lastTrain=copy(self.train)
  self._lastTest=copy(self.test)
  self._lastlogs=copy(self._logs)

  self._lastlastTrain=copy(self._lastTrain)
  self._lastlastTest=copy(self._lastTest)
  self._lastlastlogs=copy(self._lastlogs)

  force=kwargs.pop("force",None)

  rep=func(self,*args, **kwargs)

  argss= inspect.getcallargs(func,self, *args, **kwargs)
  del argss["self"]
  argss=["{}={}".format(i,"\""+j+"\"" if isinstance(j,str) else j) for i,j in argss.items()]
  self._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) ,force=force)
  return rep

def saveLast2_(self,func,*args,**kwargs):
  realSelf=kwargs.pop("realSelf",self)
  type_=kwargs.pop("type_")
  # print("saveLoad2",type_)
  if "train" in type_:
    realSelf._lastlastTrain=copy(realSelf._lastTrain)
    realSelf._lastTrain=copy(realSelf.train)
    realSelf._lastlastlogs=copy(realSelf._lastlogs)
    realSelf._lastlogs=copy(realSelf._logs)

  if "test" in type_:
    realSelf._lastlastTest=copy(realSelf._lastTest)
    realSelf._lastTest=copy(realSelf.test)
    realSelf._lastlastlogsTest=copy(realSelf._lastlogsTest)    
    realSelf._lastlogsTest=copy(realSelf._logsTest)

  force=kwargs.pop("force", None)

  realFunc=kwargs.pop("realFunc", func)

  try:
    rep=realFunc(self,*args, **kwargs)
    setattr(realSelf,type_[0],copy(getattr(self,"_data")))
    kwargs["type_"]=type_
    argss= inspect.getcallargs(func,self, *args, **kwargs)
    del argss["self"]
    argss=["{}={}".format(i,correc("\""+j+"\"" if isinstance(j,str) else j)) for i,j in argss.items()]
    realSelf._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) ,force=force,type_=type_)
  except Exception as e:
    realSelf.back_one(type_=type_)
    if hasattr(realSelf,"errors"):
      if realSelf.errors == "ignore":
        return realSelf
    raise e
  return realSelf

def saveLast3_(self,func,*args,**kwargs):
  realSelf=kwargs.pop("realSelf",self)
  type_=kwargs.pop("type_")
  # typeX_=kwargs.pop("typeX_",None)


  if "train" in type_:
    realSelf._lastlastTrain=copy(realSelf._lastTrain)
    realSelf._lastTrain=copy(realSelf.train)
    realSelf._lastlastlogs=copy(realSelf._lastlogs)
    realSelf._lastlogs=copy(realSelf._logs)

  if "test" in type_:
    realSelf._lastlastTest=copy(realSelf._lastTest)
    realSelf._lastTest=copy(realSelf.test)
    realSelf._lastlastlogsTest=copy(realSelf._lastlogsTest)    
    realSelf._lastlogsTest=copy(realSelf._logsTest)


  force=kwargs.pop("force",None)

  realFunc=kwargs.pop("realFunc",func)

  # if typeX_ is not None
  doo={i:getattr(self,i) for i in type_}

  # print(self,args,kwargs)
  try:
    rep=realFunc(self, *args, **kwargs)
    for j in type_:
      if func.__name__=="outliers" and j=="test":
        setattr(self, j, copy(doo[j]))
        continue
      setattr(realSelf, j, copy(getattr(self,j)))
      kwargs["type_"]=[j]
      argss= inspect.getcallargs(func,self, *args, **kwargs)
      del argss["self"]
      argss=["{}={}".format(k,correc("\""+w+"\"" if isinstance(w,str) else w)) for k,w in argss.items()]
      realSelf._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) ,force=force,type_=j)
  except Exception as e:
    realSelf.back_one(type_=type_)
    if hasattr(realSelf,"errors"):
      if realSelf.errors == "ignore":
        return realSelf
    raise e
  return realSelf
def saveLast(func):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      return saveLast_(self,func,*args,**kwargs)
  return with_logging
  
# def addCustomFunc2(self,func):
#   @wraps(func)
#   def with_logging(*args, **kwargs):
#       return saveLast_(self,func,*args,**kwargs)
#   return with_logging
class Speedml2(Speedml):
  def eda2(self):
      rep=self.eda()
      rep= (rep.T >> df.select(X.Shape,df.columns_from(1),df.everything())).T
      rep.Observations=rep.Observations.apply(lambda a:a.replace("feature.","prep.").replace("plot.","viz."))
      return rep
# Speedml2.eda2=eda2
    # def __init__(self,train, test, target, uid=None):
        # super().__init__(train,test,target,uid)
        # self.init(train,test,target)
# Speedml2.__init__=init2

import sys
import inspect

class My_Context_special(object):
    def __init__(self,mode=0,prep=None,fnExit=None):
        """
        if mode = 0, proceed as normal
        if mode = 1, do not execute block
        """
        self.mode=mode
        self._prep=prep
        self.fnExit=fnExit
    def __enter__(self):
        if self.mode==1:
          sys.settrace(lambda *args, **keys: None)
          frame = sys._getframe(1)
          frame.f_trace = self.trace
        else:
          return self._prep
    def trace(self, frame, event, arg):
        #print(frame.f_lasti)
        raise Exception
    def __exit__(self, type, value, traceback):
        #print('Exiting context ...')
        if self.fnExit is not None:
          self.fnExit()
        return True

class Speedml3:

    def __init__(self, train, test, target, uid=None, mode="df",reload_=None):
        # super().__init__(train,test,target,uid)
        # print("speedml3 create")
        self._Speedml=Speedml2(copy(train),copy(test),target)
        self.init(copy(train),copy(test),target, uid)
        self._snapshots = {}
        self._initial_Train=copy(train)
        self._initial_Test=copy(test)
        self.mode=mode
        self.reload_=reload_


    # def _resetTo(train,test):
    #   target=self.target
    #   self._initial_Train=copy(train)
    #   self._initial_Test=copy(test)
    #   self._Speedml=Speedml2(copy(train),copy(test),target)
    #   self.init(copy(train),copy(test),target)

    @staticmethod
    def Clone(self):
      return Speedml3(self.train ,self.test,self.target,self.uid, self.mode)
      # Speedml3( self.train, self.test, self.target, self.uid, self.mode)


    @property
    def mode(self):
      return self._mode
    
    @mode.setter
    def mode(self,mod):
      if mod not in ["df","logs"]:
          raise Exception("mode must be in df or logs")
      self._mode=mod

    def init(self,Train,Test,target, uid=None,type_=["train","test"]):
        self.uid=uid
        if "train" in type_:
          self._lastTrain=None
          self._lastlastTrain=None
          self._logs = []
          self._lastlogs=None
          self._lastlastlogs=None
          self.train=copy(Train)

        if "test" in type_:
          self._lastTest=None
          self._lastlastTest=None
          self._logsTest = []
          self._lastlogsTest=None
          self._lastlastlogsTest=None
          self.test=copy(Test)
        self.target=target


    def haveSnapshot(self, name):
       return name in self._snapshots
    #_______________SNAP_____________________
    def snapshot(self, name):
        snapshot = {
            "logs": copy(self._logs),
            "logsTest": copy(self._logsTest)#,

            # "logsLast": copy(self._lastlogs),
            # "logsTestLast": copy(self._lastlogsTest),

            # "logsLastLast": copy(self._lastlastlogs),
            # "logsTestLastLast": copy(self._lastlastlogsTest)

          }
        if self.mode == "df":
          snapshot.update({
            "dataTrain": copy(self.train),
            "dataTest": copy(self.test)#,
            # "dataTrainLast": copy(self._lastTrain),
            # "dataTrainLastLast": copy(self._lastlastTrain),
            # "dataTestLast": copy(self._lastTrain),
            # "dataTestLastLast": copy(self._lastlastTrain)  
          })
        self._snapshots[name] = snapshot

    def use_or_get_snapshot(self, name, force=False):
        if name not in self._snapshots:
            warnings.warn(f"""
              {name} not in snapshots
              """)
            self.snapshot(name)
            warnings.warn(f"""
              snapshot took with name: {name} 
              """)
        # else:
        else:
          # warnings.warn(f"""
          #     snapshot with name : {name} exist
          #     """)
          # if force:
            warnings.warn(f"""
              use snapshot with name: {name}
            """)
            self.use_snapshot(name)
          # else:
              # warnings.warn(f"""
              #   force=False so snapshot with name: {name} is not use again
              # """)
    def use_snapshot(self, name, type_=["train","test"]):
        if name not in self._snapshots:
          warnings.warn("""
            name not in snapshots
            """)
          return
        if "train" in type_:
          if self.mode == "df":
            self.train = self._snapshots[name]["dataTrain"]
          self._logs = self._snapshots[name]["logs"]
          # self._lastlogs = self._snapshots[name]["logsLast"]
          # self._lastlastlogs = self._snapshots[name]["logsLastLast"]

        if "test" in type_:
          if self.mode == "df":
            self.test = self._snapshots[name]["dataTest"]
          self._logsTest = self._snapshots[name]["logsTest"]
          # self._lastlogsTest = self._snapshots[name]["logsTestLast"]
          # self._lastlastlogsTest = self._snapshots[name]["logsTestLastLast"]

        if self.mode == "logs":
          self.train=self._initial_Train
          self.test=self._initial_Test
          d=Speedml3.Clone(self)
          # d._logs=self._logs
          # d._logsTest=self._logsTest
          d.execLogs(self._logs,self._logsTest)
          self.train = d.train
          self.test=d.test

          self._lastTrain=d._lastTrain
          self._lastTest=d._lastTest
          self._lastlastTrain=d._lastlastTrain
          self._lastlastTest =d._lastlastTest

          self._lastlogs = d._lastlogs
          self._lastlogsTest =d._lastlogsTest

          self._lastlastlogs = d._lastlastlogs
          self._lastlastlogsTest = d._lastlastlogsTest


          # self.__class__._execLogs2(d, self._logs, "")
          # self.execLogs()
          # self._execLogs([self._lastlogs,self._lastlastlogs],[self._lastlogsTest,self._lastlastlogsTest])
        
        print("snapshot loaded")

    #________________back_____________________
    def back_initial_data(self, type_=["train","test"]):
        self.init(self._initial_Train,self._initial_Test,self.target,type_=type_)

    def back_one(self,type_=["train","test"]):
        if "train" in type_:
          self.train=copy(self._lastTrain)
          self._logs=copy(self._lastlogs)
          self._lastTrain=copy(self._lastlastTrain)
          self._lastlogs=copy(self._lastlastlogs)

        if "test" in type_:
          self.test=copy(self._lastTest)
          self._logsTest=copy(self._lastlogsTest)
          self._lastTest=copy(self._lastlastTest)
          self._lastlogsTest=copy(self._lastlastlogsTest)

    #_______________LOG_________________________
    def _log(self, string,type_=["train"],force=False):
        if isinstance(type_,list):
          for i in type_:
            self._log(string,i,force)
          return
        et="" if type_=="train" else "Test"
        lg=self._logs if type_=="train" else self._logsTest
        if string in lg and not force:
          raise Exception(f"""
                _log{et}: {string} already in logs, if you want to force, add force=True""")
        lg.append(string)



    def prepSnapshot(self,name,reload=True):
      if name in self._snapshots:
        return My_Context_special(mode=1,fnExit= lambda: self.use_snapshot(name))
      return My_Context_special(mode=0,prep=self,fnExit= lambda: self.snapshot(name))

    @staticmethod
    def _execLogs2(self, logs, name):
      if logs is not None: 
        fself=Speedml3.Clone(self)
        for i in logs:
          exec(i,dict(self=fself))
        return getattr(fself,name)
      else:
        return logs

    def addCustomFunction(self,func, fn=None, type_="Dora"):
      if type_=="Dora":
        from dora_study import Dora
        Dora.addCustomFunction(func)
        fn=func.__name__ if fn is None else fn
        funci=Dora._CUSTOMS[fn]
        addMethodsFromDoraName(funci,fn,False)
        if self.reload_ is not None:
          self.reload_(funci,fn,type_)
      else:
        raise NotImplemented(f"{type_} not implented")


    # def _execLogs(self, logs, logsTest):
      # _lastlogs, _lastlastlogs = logs
      # _lastlogsTest, _lastlastlogsTest = logsTest

      # self._lastTrain=self.__class__._execLogs2(self, _lastlogs, "_lastlogs")
      # self._lastlastTrain=self.__class__._execLogs2(self, _lastlastlogs, "_lastlastlogs")
      # self._lastTest=self.__class__._execLogs2(self, _lastlogsTest, "_lastlogsTest")
      # self._lastlastTest=self.__class__._execLogs2(self, _lastlastlogsTest, "_lastlastlogsTest")

    def execLogs(self,_logs =[],_logsTest=[], lims=[None,None], type_=["train","test"]):
      if not isinstance(lims,list):
        raise Exception('lim must be list')

      if "train" in type_:
        for i in _logs:
          exec(i,dict(self=self))

      if "test" in type_:
        for i in _logsTest:
          exec(i,dict(self=self))

    #__________________IMPORT/EXPORT_________________
    def __getstate__(self):
      rep = self.__dict__
      return rep

    def __setstate__(self,d):
      self.__dict__ = d
      return 


def create_speedML(self,reload_=None):
    Train=copy(self.dataTrain.get())
    Test=copy(self.dataTest.get())
    target=self.dataTrain.y.name
    return Speedml3(Train,Test,target,reload_=reload_)

#______JUPYTER NOTEBOOK__SPECIAL_FUNC___
def _ipython_display_(self, **kwargs):
    print("Train : ",np.shape(self.train),"\nTest :",np.shape(self.test))

def make_fun(name,parameters):
    # print(parameters)
    exec("def {}({}): pass".format(name,', '.join(parameters)))
    return locals()[name]
def getVarInFn(sign):
    return ["*"+i.name if i.kind.name == "VAR_POSITIONAL" else "**"+i.name for i in list(sign.parameters.values()) if i.kind.name in ["VAR_KEYWORD","VAR_POSITIONAL"]]
def correc(l):
    # print(l,type(l))
    if type(l)==type:
        # print(l.__name__)
        return l.__name__
    return l

def getNotVarInFn(sign):
    return [f"{i.name}={correc(i.default)}" if i.default != inspect._empty else i.name for i in list(sign.parameters.values()) if i.kind.name not in ["VAR_KEYWORD","VAR_POSITIONAL"]]


# def addMethodsFromSpeedML2():
#     from speedml import Feature
#     fd=Feature.__dict__
#     n=[i for i in list(fd.keys()) if not i.startswith("_")] 
#     for i in n:
#         setattr(Speedml2,"_"+i,fd[i])
# addMethodsFromSpeedML2()

def saveLastSpeedML(func,realFunc):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      type_=kwargs.pop("type_",["train","test"])
      # d=StudyClass(_data=getattr(self,type_),_output=getattr(self,"target"))
      # d._data=getattr(self,type_)
      # d._output==getattr(self,"target")
      kwargs["realFunc"]=realFunc
      kwargs["realSelf"]=self
      kwargs["type_"]=type_
      return saveLast3_(self._Speedml,func,*args,**kwargs)
  return with_logging
def addMethodsFromSpeedML():
    from speedml_study import Feature
    fd=Feature.__dict__
    n=[i for i in list(fd.keys()) if not i.startswith("_")] 
    def job(g,i,wrapped=False):
        func=g.__wrapped__ if wrapped else g
        a=get_args(func)
        u=getVarInFn(a.signature)
        uu=getNotVarInFn(a.signature)
        o=uu+["type_=['train','test']"]
        fnu=make_fun(i,o+u)
        setattr(Speedml3,i,saveLastSpeedML(fnu,func))
    for i in n:
        job(fd[i],i)
    # for i in n:
        # setattr(Speedml2,i,getattr(Speedml2,"_"+i))
addMethodsFromSpeedML()


def saveLastDora(func,realFunc):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      type_=kwargs.pop("type_",["train","test"])
      # print(type_)
      for i in type_:
        # print(i)
        copyE=kwargs.copy()
        copyArgs=list(args).copy()
        d=StudyClass(_data=getattr(self,i),_output=getattr(self,"target"))
        # d._data=getattr(self,type_)
        # d._output==getattr(self,"target")
        copyE["realFunc"]=realFunc
        copyE["realSelf"]=self
        copyE["type_"]=[i]
        saveLast2_(d,func,*copyArgs,**copyE)
      return self
  return with_logging

def addMethodsFromDoraName(func,name,wrapped=True):
  def job(g,i,wrapped=True):
        func=g.__wrapped__ if wrapped else g
        a=get_args(func)
        u=getVarInFn(a.signature)
        uu=getNotVarInFn(a.signature)
        o=uu+["type_=['train','test']"]
        fnu=make_fun(i,o+u)
        setattr(Speedml3,i,saveLastDora(fnu,func))
  job(func,name,wrapped)
def addMethodsFromDora():
    from dora_study import Dora
    fd=Dora.__dict__
    n=[i  for i,j in Dora.__dict__.items() if not i.startswith("_") and i not in ["plot_feature","explore"] and type(j)!=classmethod and hasattr(j,"__wrapped__")] 
    for i in n:
        addMethodsFromDoraName(fd[i],i)
    fd=Dora._CUSTOMS
    for i in fd:
        addMethodsFromDoraName(fd[i],i,False)
addMethodsFromDora()
