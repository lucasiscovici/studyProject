def load_extension(self, module_str):
        """Load an IPython extension by its module name.
        Returns the string "already loaded" if the extension is already loaded,
        "no load function" if the module doesn't have a load_ipython_extension
        function, or None if it succeeded.
        """
        if module_str in self.loaded:
            return "already loaded"

        from IPython.utils.syspathcontext import prepended_to_syspath

        with self.shell.builtin_trap:
            if module_str not in sys.modules:
                with prepended_to_syspath(self.ipython_extension_dir):
                    __import__(module_str)
            mod = sys.modules[module_str]
            if self._call_load_ipython_extension(mod):
                self.loaded.add(module_str)
            else:
                return "no load function"