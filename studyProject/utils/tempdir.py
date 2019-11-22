import tempfile
import shutil
class TMP_DIR:
	def __init__(self):
        self.i=None
    def get(self):
    	self.i=tempfile.mkdtemp()
    	return i
    def delete(self):
        #print(self.i)
        shutil.rmtree(self.i)
# ... do stuff with dirpath
