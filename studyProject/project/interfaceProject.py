from interface import implements, Interface

class IProject(Interface):

    def begin(self):
        pass

    def check(self):
        pass

    # def finalize(self, x, y):
    #     pass
    def getProprocessDataFromProjectFn(self):
        pass

    def setDataTrainTest(self,X_train=None,y_train=None,
                              X_test=None,y_test=None,
                              namesY=None,id_=None):
        pass

    def proprocessDataFromProject(self,fn=None,force=False):
        pass

    def getProject(self):
        pass

    def setProject(self,p):
        pass

    def getIdData(self):
        pass

    def setIdData(self,i):
        pass