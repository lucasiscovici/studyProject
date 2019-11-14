import numpy as np
def format_perc(o,roundVal=2):
    s=None
    if np.ndim(o) != 1:
        s=np.shape(o)
        o=o.flatten()
    rep=["{}%".format(np.round(i*100,roundVal)) for i in o]
    if s is not None:
        return np.reshape(rep,s)