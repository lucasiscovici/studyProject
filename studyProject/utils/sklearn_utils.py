import sklearn
from sklearn.model_selection import check_cv
def get_metric(me):
    if hasattr(sklearn.metrics,me):
        return getattr(sklearn.metrics,me)
    elif hasattr(sklearn.metrics,me+"_score"):
        return getattr(sklearn.metrics,me+"_score")
    raise KeyError(me)

class check_cv2:
    def __init__(self,cv_=3,classifier=True,random_state=42,shuffle=True):
        self.cv_=cv_
        self.classifier=classifier
        self.random_state=random_state
        self.shuffle=shuffle
        #self.splited=self.split()
    def split(self,X,y,*args):
        e=check_cv(self.cv_,y,classifier=self.classifier)
        e.shuffle=self.shuffle
        e.random_state=self.random_state
        return list(e.split(X,y))