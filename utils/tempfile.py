import tempfile
import os
class TMP_FILE:
    def __init__(self):
        self.i=None
    def get_filename(self,ext="png"):
        if self.i is not None:
           self.delete()
        _,self.i=tempfile.mkstemp(suffix='.'+ext)
        return self.i
    def delete(self):
        #print(self.i)
        os.remove(self.i)