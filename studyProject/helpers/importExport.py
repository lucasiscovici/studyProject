from ..utils import SaveLoad, to_camel_case
from ..utils.save_load import COMPRESSION_TYPES 
from ..base import factoryCls
import os
def importFromFile(path,pathHere=os.getcwd(),delim="/",dirStudyFiles="__studyFiles",inferDirs=True):
	splitted=path.split(delim)
	spili=splitted[-1]
	argus=spili.split(".")
	if len(argus)<3:
		raise KeyError("{} n'est pas du bon type pour proceder a une inference".format(path))

	ext=argus[-1]
	if ext not in COMPRESSION_TYPES:
		raise KeyError("{} n'est pas du bon type pour proceder a une inference".format(ext))
	if ext =="None":
		ext=None
	exp=argus[-2]

	if exp != "EXP":
		raise KeyError("{} n'est pas du bon type pour proceder a une inference".format(ext))

	cls=argus[-3]

	if len(cls)<len("study_"):
		raise KeyError("{} n'est pas du bon type pour proceder a une inference".format(cls))

	cls0=to_camel_case(cls[len("study_"):])
	if inferDirs:
		path=dirStudyFiles+delim+cls+"_EXP"+delim+path
	cls0=cls0[0].upper()+cls0[1:]

	if cls0 not in factoryCls._classes:
		raise KeyError("{} n'est pas du bon type pour proceder a une inference".format(cls0))

	clsO=factoryCls._classes[cls0]

	path=path if pathHere == "" else pathHere+delim+path
	return clsO.Import(path,noDefaults=True,path="",addExtension=False,loadArgs=dict(compression=ext))

