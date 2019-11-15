class LogProgress:
    def __init__(self,sequence, every=None, size=None, name='Items',generatorExitAsSuccess=False):
        self.sequence=sequence
        self.every=every
        self.size=size
        self.name=name
        self.generatorExitAsSuccess=generatorExitAsSuccess

    def log_progress(self):
        from ipywidgets import IntProgress, HTML, VBox
        from IPython.display import display
        sequence=self.sequence
        every=self.every
        size=self.size
        name=self.name
        generatorExitAsSuccess=self.generatorExitAsSuccess
        def fin():
            progress.bar_style = 'success'
            progress.value = index
            label.value = "{name}: {index} / {size}".format(
                name=name,
                index=str(index or '?'),
                size=size
            )
        def err():
            progress.bar_style = 'danger'
        is_iterator = False
        if size is None:
            try:
                size = len(sequence)
            except TypeError:
                is_iterator = True
        if size is not None:
            if every is None:
                if size <= 200:
                    every = 1
                else:
                    every = int(size / 200)     # every 0.5%
        else:
            assert every is not None, 'sequence is iterator, set every'

        if is_iterator:
            progress = IntProgress(min=0, max=1, value=1)
            progress.bar_style = 'info'
        else:
            progress = IntProgress(min=0, max=size, value=0)
        label = HTML()
        box = VBox(children=[label, progress])
        display(box)

        index = 0
        try:
            for index, record in enumerate(sequence, 1):
                # sequence=self.sequence
                # every=self.every
                size=self.size
                name=self.name
                # generatorExitAsSuccess=self.generatorExitAsSuccess
                if index == 1 or index % every == 0:
                    if is_iterator:
                        label.value = '{name}: {index} / ?'.format(
                            name=name,
                            index=index
                        )
                    else:
                        progress.value = index
                        label.value = u'{name}: {index} / {size}'.format(
                            name=name,
                            index=index,
                            size=size
                        )
                yield record
        except GeneratorExit:
            if generatorExitAsSuccess:
                fin()
            else:
                err()
                raise
        except:
            err()
            raise
        else:
            fin()

class ProgressBarCalled:
    def __init__(self,*,seq=None,total=None,name="Items",generatorExitAsSuccess=False,
                fnCalled=lambda args,xargs,pb:None):
        if seq is None:
            if total is None:
                raise Exception("seq and total None")
            seq=range(0,total)
        self.fnCalled=fnCalled
        self.pb=LogProgress(seq,size=total,name=name,generatorExitAsSuccess=generatorExitAsSuccess)
        self.pb_it=self.pb.log_progress()

    def __call__(self,*args,**xargs):
        self.fnCalled(args,xargs,self.pb)
        next(self.pb_it)
