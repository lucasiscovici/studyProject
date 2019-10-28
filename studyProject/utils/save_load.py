from . import compress_pickle
import warnings
from . import showWarningsTmp
import os
DEFAULT_COMPRESSION="bz2"
class SaveLoad:
    @staticmethod
    def load(name,n="rb", compression=DEFAULT_COMPRESSION,set_default_extension=False,chut=True,**xargs):
        #return dill.load(open(name,n))
        ty="\n[SaveLoad load] Loading {}".format(name)
        # warnings.warn(ty)
        # print("id")
        if not chut:
            with showWarningsTmp:
                warnings.warn(ty)
        rep= compress_pickle.load(name,compression=compression,
                                    set_default_extension=set_default_extension,**xargs)
        return rep
    @staticmethod
    def save(selfo,name,n="wb", compression=DEFAULT_COMPRESSION,set_default_extension=False,chut=True,preventError=True,**xargs):
        #return dill.dump(selfo,open(name,n),)
        if not chut:
            ty="\n[SaveLoad load] Saving {}".format(name)
            with showWarningsTmp:
                warnings.warn(ty)
        if preventError:
            if os.path.isfile(name):
                try:
                    old=SaveLoad.load(name,chut=True)
                except:
                    preventError=False
                    if not chut:
                        ty="\n[SaveLoad load] No prevent possible pour {}".format(name)
                        with showWarningsTmp:
                            warnings.warn(ty)
            else:
                preventError=False
        try:
            rep=compress_pickle.dump(selfo,name,compression=compression,
                                        set_default_extension=set_default_extension,**xargs)
        except Exception as e:
            if preventError:
                SaveLoad.save(old,name)
                if not chut:
                    ty="[SaveLoad load] Saving OLD FileName {}".format(name)
                    with showWarningsTmp:
                        warnings.warn(ty)
                raise e
