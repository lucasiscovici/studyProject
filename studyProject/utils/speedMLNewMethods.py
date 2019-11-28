from speedml import Speedml
from studyPipe import df, X
import numpy as np
import pandas as pd
from . import get_args, StudyClass
import inspect
if not hasattr(Speedml,"__init__base"): Speedml.__init__base=Speedml.__init__
def init2(self,train, test, target, uid=None):
    from speedml import Plot, Feature, Model, Xgb, Base
    spd=self
    spd._setup_environment()
    Base.target = target
    Base.train=train
    Base.test=test
    uid=None

    if not Base.train is None and not Base.test is None:
#         if uid:
#             Base.uid = Base.test.pop(uid)
#             Base.train = Base.train.drop([uid], axis=1)
        spd.plot = Plot()
        spd.feature = Feature()
        spd.xgb = Xgb()
        spd.model = Model()

        spd.np = np
        spd.pd = pd
    else:
        print('ERROR: SpeedML can only process .csv and .json file extensions.')
    #return spd
Speedml.__init__=init2
import inspect
from functools import wraps

def has_method(o, name):
    return name in dir(o)

def saveLast_(self,func,*args,**kwargs):
  self._lastTrain=self.train.copy()
  self._lastTest=self.test.copy()
  self._lastlogs=self._logs.copy()

  self._lastlastTrain=self._lastTrain.copy()
  self._lastlastTest=self._lastTest.copy()
  self._lastlastlogs=self._lastlogs.copy()

  force=kwargs.pop("force",None)

  rep=func(self,*args, **kwargs)

  argss= inspect.getcallargs(func,self, *args, **kwargs)
  del argss["self"]
  argss=["{}={}".format(i,"\""+j+"\"" if isinstance(j,str) else j) for i,j in argss.items()]
  self._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) ,force=force)
  return rep

def saveLast2_(self,func,*args,**kwargs):
  realSelf=kwargs.pop("realSelf",self)
  realSelf._lastTrain=realSelf.train.copy()
  realSelf._lastTest=realSelf.test.copy()
  realSelf._lastlogs=realSelf._logs.copy()

  realSelf._lastlastTrain=realSelf._lastTrain.copy()
  realSelf._lastlastTest=realSelf._lastTest.copy()
  realSelf._lastlastlogs=realSelf._lastlogs.copy()

  force=kwargs.pop("force",None)

  type_=kwargs.pop("type_")
  realFunc=kwargs.pop("realFunc",func)


  rep=realFunc(self,*args, **kwargs)
  setattr(realSelf,type_,getattr(self,"_data"))

  argss= inspect.getcallargs(func,self, *args, **kwargs)
  del argss["self"]
  argss=["{}={}".format(i,correc("\""+j+"\"" if isinstance(j,str) else j)) for i,j in argss.items()]
  realSelf._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) ,force=force,type_=type_)
  return realSelf

def saveLast3_(self,func,*args,**kwargs):
  realSelf=kwargs.pop("realSelf",self)
  type_=kwargs.pop("type_")
  # typeX_=kwargs.pop("typeX_",None)
  if "train" in type_:
    realSelf._lastlastTrain=realSelf._lastTrain.copy()
    realSelf._lastTrain=realSelf.train.copy()
    realSelf._lastlastlogs=realSelf._lastlogs.copy()
    realSelf._lastlogs=realSelf._logs.copy()

  if "test" in type_:
    realSelf._lastlastTest=realSelf._lastTest.copy()
    realSelf._lastTest=realSelf.test.copy()
    realSelf._lastlastlogsTest=realSelf._lastlogsTest.copy()    
    realSelf._lastlogsTest=realSelf._logsTest.copy()


  force=kwargs.pop("force",None)

  realFunc=kwargs.pop("realFunc",func)

  # if typeX_ is not None
  doo={i:getattr(self,i) for i in type_}

  rep=realFunc(self,*args, **kwargs)
  for i in type_:
    setattr(realSelf,i,getattr(rep,i))

  for i in type_:
    setattr(realSelf,i,getattr(self,doo[i]))

  # kwargs["type_"]=type_
  argss= inspect.getcallargs(func,self, *args, **kwargs)
  del argss["self"]
  argss=["{}={}".format(i,correc("\""+j+"\"" if isinstance(j,str) else j)) for i,j in argss.items()]
  realSelf._log( "self.{}({})".format( func.__name__, ", ".join(argss) ) ,force=force,type_=type_)
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
class Speedml2:

    def __init__(self,train, test, target, uid=None):
        # super().__init__(train,test,target,uid)
        self._Speedml=Speedml(train,test,target)
        self.init(train,test,target)
        self._snapshots = {}
        self._initial_Train=train.copy()
        self._initial_Test=test.copy()


    def init(self,Train,Test,target,type_=["train","test"]):

        if "train" in type_:
          self._lastTrain=None
          self._lastlastTrain=None
          self._logs = []
          self._lastlogs=None
          self._lastlastlogs=None
          self.train=Train

        if "test" in type_:
          self._lastTest=None
          self._lastlastTest=None
          self._logsTest = []
          self._lastlogsTest=None
          self._lastlastlogsTest=None
          self.test=Test


    #_______________SNAP_____________________
    def snapshot(self, name):
        snapshot = {
          "dataTrain": self.train.copy(),
          "dataTest": self.test.copy(),
          "logs": self._logs.copy(),
          "logsTest": self._logsTest.copy()
        }
        self._snapshots[name] = snapshot

    def use_snapshot(self, name, type_=["train","test"]):
        if "train" in type_:
          self.train = self._snapshots[name]["dataTrain"]
          self._logs = self._snapshots[name]["logs"]

        if "test" in type_:
          self.test = self._snapshots[name]["dataTest"]
          self._logsTest = self._snapshots[name]["logsTest"]

    #________________back_____________________
    def back_initial_data(self, type_=["train","test"]):
        self.init(self._initial_Train,self._initial_Test,self.target,type_=type_)

    def back_one(self,type_=["train","test"]):
        if "train" in type_:
          self.train=self._lastTrain.copy()
          self._logs=self._lastlogs.copy()
          self._lastTrain=self._lastlastTrain.copy()
          self._lastlogs=self._lastlastlogs.copy()

        if "test" in type_:
          self.test=self._lastTest.copy()
          self._logsTest=self._lastlogsTest.copy()
          self._lastTest=self._lastlastTest.copy()
          self._lastlogsTest=self._lastlastlogsTest.copy()
    #_______________LOG_________________________
    def _log(self, string,type_="train",force=False):
      et="" if type_=="train" else "Test"
      lg=self._logs if type_=="train" else self._logsTest
        if string in lg and not force:
          raise Exception(f"""
                _log{et}: {string} already in logs, if you want to force, add force=True""")
        lg.append(string)

def create_speedML(self):
    Train=self.dataTrain.get()
    Test=self.dataTest.get()
    target=self.dataTrain.y.name
    return Speedml2(Train,Test,target)

def eda2(self):
    rep=self.eda()
    rep= (rep.T >> df.select(X.Shape,df.columns_from(1),df.everything())).T
    rep.Observations=rep.Observations.apply(lambda a:a.replace("feature.","prep.").replace("plot.","viz."))
    return rep
Speedml2.eda2=eda2
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
      type_=kwargs.pop("type_",["train"])
      d=StudyClass(_data=getattr(self,type_),_output=getattr(self,"target"))
      # d._data=getattr(self,type_)
      # d._output==getattr(self,"target")
      kwargs["realFunc"]=realFunc
      kwargs["realSelf"]=self
      kwargs["type_"]=type_
      return saveLast3_(d,func,*args,**kwargs)
  return with_logging
def addMethodsFromSpeedML():
    from speedml import Feature
    fd=Feature.__dict__
    n=[i for i in list(fd.keys()) if not i.startswith("_")] 
    def job(g,i,wrapped=False):
        func=g.__wrapped__ if wrapped else g
        a=get_args(func)
        u=getVarInFn(a.signature)
        uu=getNotVarInFn(a.signature)
        o=uu+["type_=['train']"]
        fnu=make_fun(i,o+u)
        setattr(Speedml2,i,saveLastSpeedML(fnu,func))
    for i in n:
        job(fd[i],i)
    # for i in n:
        # setattr(Speedml2,i,getattr(Speedml2,"_"+i))
addMethodsFromSpeedML()


def saveLastDora(func,realFunc):
  @wraps(func)
  def with_logging(self,*args, **kwargs):
      type_=kwargs.pop("type_","train")
      d=StudyClass(_data=getattr(self,type_),_output=getattr(self,"target"))
      # d._data=getattr(self,type_)
      # d._output==getattr(self,"target")
      kwargs["realFunc"]=realFunc
      kwargs["realSelf"]=self
      kwargs["type_"]=type_
      return saveLast2_(d,func,*args,**kwargs)
  return with_logging

def addMethodsFromDora():
    from dora_study import Dora
    fd=Dora.__dict__
    n=[i  for i,j in Dora.__dict__.items() if not i.startswith("_") and i not in ["plot_feature","explore"] and type(j)!=classmethod and hasattr(j,"__wrapped__")] 
    def job(g,i,wrapped=True):
        func=g.__wrapped__ if wrapped else g
        a=get_args(func)
        u=getVarInFn(a.signature)
        uu=getNotVarInFn(a.signature)
        o=uu+["type_='train'"]
        fnu=make_fun(i,o+u)
        setattr(Speedml2,i,saveLastDora(fnu,func))
    for i in n:
        job(fd[i],i)
    fd=Dora._CUSTOMS
    # print(fd)
    for i in fd:
        job(fd[i],i,False)
addMethodsFromDora()