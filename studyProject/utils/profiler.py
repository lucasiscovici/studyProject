import profile
import pstats 
from . import TMP_FILE
def profile_that(blabla,gl=lambda : globals(),l=lambda:locals()):
	f=TMP_FILE()
	filename=f.get_filename("txt")
	rep=profile.runctx(blabla,gl(),l(),filename=filename)
	stats_=pstats.Stats(filename)
	f.delete()
	return stats_