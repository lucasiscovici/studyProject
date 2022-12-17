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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f)
    os.chdir(ar)
    return where