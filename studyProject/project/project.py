from ..base import Base, DatasSupervise, BaseSupervise
from ..utils import getStaticMethodFromObj, ifelse, BeautifulDico, SaveLoad, mapl,takeInObjIfInArr, StudyDict, StudyClass
import os
from . import IProject
from typing import *
from interface import implements
class StudyProject(Base):

    DEFAULT_REP="study_project"
    DEFAULT_EXT=".studyProject"
    EXPORTABLE=["studies","curr","data"]
    EXPORTABLE_ARGS=dict(underscore=True)
    def __init__(self,ID=None,studies:Dict[str,BaseSupervise]=None,
                    curr=None,data:Dict[str,DatasSupervise]=None):
        super().__init__(ID)
        self._studies={} if studies is None else studies
        self._curr=curr
        self._data=data if data is not None else {}#BeautifulDico({"_ZERO":DatasSupervise.from_XY_Train_Test(None,None,None,None,None,ID="_ZERO")})
        
    def add(self,study_):
        study_2=study_
        if isinstance(study_2,implements(IProject)):
            study_2._project=self
        self._studies[study_.ID]=study_2
        self._curr=study_2.ID
        return study_2
    
    def get(self,id_=None):
        id_=ifelse(id_,id_,self._curr)
        return self._studies[id_]
    
    def getAndCurr(self,id_):
        self._curr=id_
        return self.get(od_)
        
    def addOrGetStudy(self,id_,cls,recreate=False,clone=False,deep=True,imported=False):
        def clonee(rrt):
            return getStaticMethodFromObj(rrt,"clone")(rrt,deep=deep)
        def recreatee():
            rrt=cls(ID=id_)
            if clone :
                rrt=clonee(rrt)
            res =self.add(rrt)
            return res
        def cloneStudy():
            ru=self._studies[id_] 
            ru=clonee(ru)
            self._studies[id_]=ru
            return self._studies[id_]
        if recreate:
            res=recreatee()
        else:
            if imported:
                res= cls.Import(id_)
                res = clonee(res) if clone else res
                self.add(res)
            else:
                res=ifelse(id_ in self._studies,
                          lambda:self._studies[id_] if not clone else cloneStudy() ,
                          lambda:recreatee())()
        if isinstance(res,implements(IProject)):res.check() 
        self._curr=id_
        return res
    
    @property
    def currStudy(self):
        return self.get()
        
    
    @classmethod
    def getOrCreate(cls,ID,repertoire=None,ext=None,
                    path=os.getcwd(),delim="/",
                    recreate=False,clone=False,deep=True,chut=True,**xargs):
        from .baseProject import IProject
        # repertoire = ifelse(repertoire is None,StudyProject.DEFAULT_REP,repertoire)
        # ext=ifelse(ext is None,StudyProject.DEFAULT_EXT,ext)
        def clonee(rrt):
            return getStaticMethodFromObj(rrt,"clone")(rrt,deep=deep)
        # repo=path+delim+repertoire
        # print(repo)
        if recreate:
            return StudyProject(ID)
        repo = cls.get_repertoire(repertoire)
        if not os.path.exists(repo):
            return StudyProject(ID)

        filo,ok=cls.build_ext(repo,cls.get_ext(ext),ID,
            delim=delim,recreate=recreate,chut=chut)
        if not os.path.isfile(filo):
            return StudyProject(ID)

        sl=cls.Load(filo,**xargs)
        sf={}
        for k,v_ in sl._studies.items():
            v=ifelse(clone,lambda: clonee(v_),lambda:v_)()
            if isinstance(v,implements(IProject)):
                v.begin()
                v.setProject(sl)
                #print(v.idData)
                v.setDataTrainTest(id_=v.getIdData())
                try:
                    v.proprocessDataFromProject(v.getProprocessDataFromProjectFn())
                except:
                    warnings.warn("[StudyProject getOrCreate] pb with {} when proprocessDataFromProject".format(k))
                    pass
                    #print("Error")
                    #print(inst)
                        #print(v.isProcess)
                v.check()
            sf[k]=v
        sl._studies=sf
        return sl
    



    def save(self,repertoire=None,ext=None,path=os.getcwd(),
             delim="/",returnOK=False,**xargs):
        ID=self.ID
        # repertoire = ifelse(repertoire is None,StudyProject.DEFAULT_REP,repertoire)
        # ext=ifelse(ext is None,StudyProject.DEFAULT_EXT,ext)
        # repo=path+delim+repertoire
        # if not os.path.exists(repo):
        #     os.makedirs(repo)
        # filo=repo+delim+ID+ext
        sl=StudyProject.clone(self,deep=False)
        # sl=StudyProject.clone(self,deep=True)
        ff={}
        for k,v_ in sl._studies.items():
            v=v_
            # print(isinstance(v_,implements(IProject)))
            if isinstance(v_,implements(IProject)):
                # v__=v.project
                # v=v_.clone(withoutProject=True)
                li=v.getIdData()
                # v._project=v__
                v._datas={}
                # v.setDataTrainTest(id_="_ZERO")
                v.setIdData(li)
                v.setProject(v.ID)

                # print(v)
            ff[k]=v

        # print(ff)
        sl._studies=ff
        # copye=StudyProject.clone(sl,deep=True)
        # copye._studies=ff
        # sl=copye
        if returnOK:
            return sl
        else:
            self.__class__.Save(sl,ID,repertoire=repertoire,ext=ext,path=path,delim=delim,**xargs)
            # SaveLoad.save(sl,filo,**xargs)
    
    
    
    def clone_(self,ID=None,deep=False):
        return getStaticMethodFromObj(self,"clone")(ID,deep)
        
    # @staticmethod
    # def clone(self,ID=None,deep=False):

    #     cloned=Base.clone(self,ID=ID,deep=deep)
    #     if deep:
    #         cloned._studies={k:getStaticMethodFromObj(v,"clone")(v,v.ID,deep=True) for k,v in self._studies.items()}
    #     return cloned
    
    @staticmethod
    def fromStudyProject(self,studyG):
        return studyG.clone(studyG.name)
    
    def saveDatasWithId(self,id_,X_train,y_train,X_test,y_test,namesY=None):
        self._data[id_]=DatasSupervise.from_XY_Train_Test(*[X_train,y_train,X_test,y_test,namesY,id_])

    def saveDataSupervise(self,dataSup):
        id_= dataSup.ID
        self._data[id_] =  dataSup.clone()
    @staticmethod
    def plan(fromHelp=False):
        print("PLAN Project:")
        if not fromHelp:
            print("\tCREATE Study Project:")
            print("\t\t[Name_Variable_Study_Project].getOrCreate(ID_OF_PROJECT)")
        print("\tSET Project Data (when original data are the same for multiple study):")
        print("\t\tWith X,y (train and test) and labels of y:"+"\n\t\t\t[Name_Variable_Study_Project].saveDatasWithId(ID_OF_DATA,X_train, y_train, X_test, y_test, namesY)")
        print("\tCreate or Get study inside the project:")
        print("\t\t[Name_Variable_Study_Project].addOrGetStudy(ID_STUDY,CLASS_OF_STUDY__OR__OTHER_STUDY_CLONE)")
        print("\t\t\tEx: [Name_Variable_Study_Project].addOrGetStudy('HELLO_WORLD',StudyClassifProject) ")
        print("\t\t\tEx: [Name_Variable_Study_Project].addOrGetStudy('HELLO_WORLD',[Name_Variable_Study].clone)")
        print("\tSAVE:")
        print("\t\t[Name_Variable_Study].save()")

    def help(self):
        getStaticMethodFromObj(self,"plan")(fromHelp=True)

    def __repr__(self,ind=1,onlyID=False):
        t="\t"*ind
        nt="\n"+t
        ntt=""
        ff=self._studies
        if onlyID:
            return ("[[StudyProject]"+nt+"ID : {}]").format(self.ID)
        if len(ff)>0:
            ntt=nt+"\t"+t.join(mapl(lambda a:"ID : "+a,ff))
            # print(ntt)
            rep= ("[[StudyProject]"+nt+"ID : {}"+nt+"Studies : {}]").format(self.ID,ntt)
        else:
            ntt="0"
            rep =("[[StudyProject]"+nt+"ID : {}"+nt+"Studies : {}]").format(self.ID,ntt)
        if len(self._data)>0:
            ntt=nt+"\t"+(nt+"\t").join(mapl(lambda a:"ID : "+a,self._data))
            rep= (rep[:-1]+nt+"Datas : {}]").format(ntt)
        else:
            ntt="0"
            rep =(rep[:-1]+nt+"Datas : {}]").format(ntt)
        return rep

    def export(self,save=True,*args,**xargs):
        obj=self.save(returnOK=True)
        return self.__class__.Export(obj,save=save,*args,**xargs)

    @classmethod 
    def _import(cls,loaded,clone=False):
        sf={}
        sl=loaded
        for k,v_ in sl._studies.items():
            v=ifelse(clone,lambda: clonee(v_),lambda:v_)()
            # print(v)
            if isinstance(v,implements(IProject)):
                # print(v_)
                v.begin()
                v.setProject(sl)
                # print(v.getIdData())
                v.setDataTrainTest(id_=v.getIdData())
                try:
                    v.proprocessDataFromProject(v.getProprocessDataFromProjectFn())
                except:
                    warnings.warn("[StudyProject getOrCreate] pb with {} when proprocessDataFromProject".format(k))
                    #print("Error")
                    #print(inst)
                        #print(v.isProcess)
                v.check()
            sf[k]=v
        sl._studies=sf
        return sl

    # @classmethod
    # def _export(cls,obj):
    #     obj=obj.save(returnOK=True)
    #     # rep=takeInObjIfInArr(cls.EXPORTABLE,obj)

    #     # rep={k:(v.export(save=False) if isinstance(v,Base) else v) for k,v in rep.items()}
    #     # rep["data"] = {k:v.export(save=False) for k,v in rep["data"].items()}
    #     # rep["studies"] = {k:v.export(save=False) for k,v in rep["studies"].items()}
    #     return obj
