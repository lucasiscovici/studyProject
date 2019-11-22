import tarfile
import os

def make_tarfile(archive_name, sources_dir):
    with tarfile.open(archive_name, mode='w:gz') as archive:
    	for i in sources_dir:
    		archive.add(i)
    return archive_name

def read_tarfile(archive_name,where=None):
	if where is None:
		dff=TMP_DIR()
	with tarfile.open(archive_name, mode='w:gz') as f:
		os.chdir(dff.get())
		tar.extractall()
	return dff