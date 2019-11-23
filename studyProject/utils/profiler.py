import profile
import pstats 
from . import TMP_FILE
from . import hidePrint
from . import StudyClass
def profile_that(blabla,gl=lambda : globals(),l=lambda:locals()):
    f=TMP_FILE()
    filename=f.get_filename("txt")
    rep=profile.runctx(blabla,gl(),l(),filename=filename)
    stats_=pstats.Stats(filename)
    f.delete()
    return stats_

def profile_that_snake(balblabla,hidePrint_=True,tg=False,pkg="snakeviz_study",port="6006"):
    ip=get_ipython()
    try:
        if hidePrint_ or tg:
            with hidePrint():
                ip.run_line_magic("load_ext",pkg)
        else:
            ip.run_line_magic("load_ext",pkg)
        ip.run_line_magic(pkg,"-p "+port+" "+balblabla)
    except Exception as e:
        if not tg:
            raise e

def addToPath(path,opts,o,rep=False):
    if o in opts:
        return "-"+o+" "+("" if not rep else opts[o]+" ")+path
    return path
config = StudyClass(pkg="snakeviz_study",port="6006",magicName="snakeviz")
try:
    from IPython.core.magic import Magics, magics_class, line_cell_magic
    from IPython.display import display, HTML

    @magics_class
    class profile_that_Magic(Magics):
        @line_cell_magic
        def profile_that(self, lineX, cell=None):
            global config
            ip=get_ipython()
            opts, line = self.parse_options(lineX, "tqp:H:f:a","new-tab","auto", posix=False)
            port = config.port if "p" not in opts else opts["p"]
            line = "-p "+port+" "+line
            if "a" in opts or "auto" in opts:
                if "f" in opts :
                    del opts["f"]
                f=TMP_FILE()
                line= "-f "+(f.get_filename("html"))+" "
            line=addToPath(line,opts,"t")
            line=addToPath(line,opts,"q")
            line=addToPath(line,opts,"H",True)
            line=addToPath(line,opts,"f",True)

            pkg=config.pkg
            with hidePrint():
                ip.run_line_magic("load_ext",pkg)
            if cell:
                ip.run_cell_magic(config.magicName, line, cell)
            else:
                ip.run_line_magic(config.magicName, line)
            if "a" in opts or "auto" in opts:
                f.delete()

except ImportError:
    pass

def load_ipython_extension(ipython):
        ipython.register_magics(profile_that_Magic)

p=get_ipython()
load_ipython_extension(p.extension_manager.shell)