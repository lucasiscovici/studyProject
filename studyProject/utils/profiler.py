import profile
import pstats 
from . import TMP_FILE
from . import hidePrint
def profile_that(blabla,gl=lambda : globals(),l=lambda:locals()):
    f=TMP_FILE()
    filename=f.get_filename("txt")
    rep=profile.runctx(blabla,gl(),l(),filename=filename)
    stats_=pstats.Stats(filename)
    f.delete()
    return stats_

def profile_that_snake(balblabla,hidePrint_=True,tg=False,pkg="snakeviz2",port="6006"):
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

try:
    from IPython.core.magic import Magics, magics_class, line_cell_magic
    from IPython.display import display, HTML

    @magics_class
    class profile_that_Magic(Magics):
        @line_cell_magic
        def profile_that(self, line, cell=None):
            ip=get_ipython()
            opts, line = self.parse_options(line, "p:", posix=False)
            port = "6006" if "p" not in opts else opts["p"]
            line = "-p "+port+" "+line
            pkg="snakeviz2"
            with hidePrint():
                ip.run_line_magic("load_ext",pkg)
            if cell:
                ip.run_cell_magic("snakeviz2", line, cell)
            else:
                ip.run_line_magic("snakeviz2", line)
except ImportError:
    pass

def load_ipython_extension(ipython):
        ipython.register_magics(profile_that_Magic)