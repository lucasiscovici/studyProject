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
import inspect
from functools import wraps

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
  realSelf._lastTrain=copy(realSelf.train)
  realSelf._lastTest=copy(realSelf.test)
  realSelf._lastlogs=copy(realSelf._logs)

  realSelf._lastlastTrain=copy(realSelf._lastTrain)
  realSelf._lastlastTest=copy(realSelf._lastTest)
  realSelf._lastlastlogs=copy(realSelf._lastlogs)

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
  rep=realFunc(self,*args, **kwargs)
  for i in type_:
    setattr(realSelf,i,getattr(self,i))

  for i in type_:
    setattr(self,i,doo[i])

  for j in type_:
    kwargs["type_"]=[j]
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
class Speedml2(Speedml):
  pass
    # def __init__(self,train, test, target, uid=None):
        # super().__init__(train,test,target,uid)
        # self.init(train,test,target)
Speedml2.__init__=init2
class Speedml3:

    def __init__(self,train, test, target, uid=None):
        # super().__init__(train,test,target,uid)
        self._Speedml=Speedml2(train,test,target)
        self.init(train,test,target)
        self._snapshots = {}
        self._initial_Train=copy(train)
        self._initial_Test=copy(test)


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
          "dataTrain": copy(self.train),
          "dataTest": copy(self.test),
          "logs": copy(self._logs),
          "logsTest": copy(self._logsTest)
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

def create_speedML(self):
    Train=self.dataTrain.get()
    Test=self.dataTest.get()
    target=self.dataTrain.y.name
    return Speedml3(Train,Test,target)

def eda2(self):
    rep=self.eda()
    rep= (rep.T >> df.select(X.Shape,df.columns_from(1),df.everything())).T
    rep.Observations=rep.Observations.apply(lambda a:a.replace("feature.","prep.").replace("plot.","viz."))
    return rep
Speedml3.eda2=eda2
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
    from speedml import Feature
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
        setattr(Speedml3,i,saveLastDora(fnu,func))
    for i in n:
        job(fd[i],i)
    fd=Dora._CUSTOMS
    # print(fd)
    for i in fd:
        job(fd[i],i,False)
addMethodsFromDora()