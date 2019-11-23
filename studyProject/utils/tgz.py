import tarfile
import os
from . import TMP_DIR 

def make_tarfile(archive_name, sources_dir,ext="gz"):
    # print(sources_dir)
    # print("create tarFile",archive_name)
    with tarfile.open(archive_name, mode='w:'+ext) as archive:
        ar=os.getcwd()
        for i in sources_dir:
            bi=os.path.basename(i)
            fi=os.path.dirname(i)
            os.chdir(fi)
            archive.add(bi)
        os.chdir(ar)
    return archive_name

def read_tarfile(archive_name,where=None, ext="gz"):
    # raise NotImplementedError("rff")
    if where is None:
        dff=TMP_DIR()
        where=dff.get()
    ar=os.getcwd()
    with tarfile.open(archive_name, mode='r:'+ext) as f:
    #     print(GH)
        os.chdir(where)
        f.extractall()
    os.chdir(ar)
    return where