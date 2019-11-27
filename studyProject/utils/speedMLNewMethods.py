from speedml import Speedml
from studyPipe import df, X
import numpy as np
import pandas as pd
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
    def __init__(self,train, test, target, uid=None):
        super().__init__(train,test,target,uid)
        self.init(train,test,target)


    def init(self,Train,Test,target):
        self._snapshots = {}

        self._lastTrain=None
        self._lastlastTrain=None

        self._lastTest=None
        self._lastlastTest=None

        self._logs = []
        self._lastlogs=None
        self._lastlastlogs=None

        self.train=Train
        self.test=Test

        self._initial_Train=Train.copy()
        self._initial_Test=Test.copy()

    #_______________SNAP_____________________
    def snapshot(self, name):
        snapshot = {
          "dataTrain": self.train.copy(),
          "dataTest": self.test.copy(),
          "logs": self._logs.copy()
        }
        self._snapshots[name] = snapshot

    def use_snapshot(self, name):
        self.train = self._snapshots[name]["dataTrain"]
        self.test = self._snapshots[name]["dataTest"]
        self._logs = self._snapshots[name]["logs"]

    #________________back_____________________
    def back_initial_data(self):
        self.init(self._initial_Train,self._initial_Test,self.target)

    def back_one(self):
        self.train=self._lastTrain.copy()
        self.test=self._lastTest.copy()
        self._logs=self._lastlogs.copy()
        self._lastTrain=self._lastlastTrain.copy()
        self._lastTest=self._lastlastTest.copy()
        self._lastlogs=self._lastlastlogs.copy()

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


def addMethodsFromSpeedML():
    from speedml import Feature
    fd=Feature.__dict__
    n=[i for i in list(fd.keys()) if not i.startswith("_")] 
    for i in n:
        setattr(Speedml2,i,saveLast(fd[i]))
addMethodsFromSpeedML()
