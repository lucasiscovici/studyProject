from .util2 import getPrivateAttr,flatArray, getClassName, has_method, merge, \
					randomString, uniquify, mapl, zipl, filterl, rangel, ifEmpty, \
					remove_empty_keys, ifelse, ifelseLenZero, getWarnings, setWarnings, \
					offWarnings, onWarnings, ShowWarningsTmp, HideWarningsTmp, \
					hideWarningsTmp, showWarningsTmp, newStringUniqueInDico, removeNone, \
					getStaticMethod, getStaticMethodFromObj, getStaticMethodFromCls, merge_two_dicts, ifNotNone,\
					T, F, getsourceP, takeInObjIfInArr, convertCamelToSnake, securerRepr, getAnnotationInit,\
					 merge_dicts, iterable, to_camel_case, check_names, namesEscape, listl,\
					 isPossible, isNotPossible, numpyToCatPdSeries, changeTmpObj, get_args, get_default_args,blockPrint, enablePrint, hidePrint, setattrAndReturnSelf, indexOfMinForValueInArray
from .is_ import isInt, isStr, isNumpyArr, isArr
from .struct import StudyList, StudyDict, StudyNpArray,dicoAuto, studyDico, studyList, Obj, BeautifulDico, BeautifulList, StudyClass, instanceOfType,isinstanceBase, isinstance
from .sklearn_utils import get_metric, check_cv2
from .compress_pickle import compress_pickle
from .save_load import SaveLoad
from .tempfile import TMP_FILE
from .profiler import profile_that, profile_that_snake
from .pandasNewMethods import *
from .colors import luminence, frontColorFromColorscaleAndValue, flipScale, frontColorFromColorscaleAndValues
from .completer import config_completer, createSubClassFromIPCompleter, returnCom
from .image import IMG, IMG_GRID