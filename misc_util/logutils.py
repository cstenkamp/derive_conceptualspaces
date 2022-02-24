import logging
import contextlib
import io, sys
from datetime import datetime
import warnings
from os.path import dirname, join, abspath
from os import sep

from misc_util.object_wrapper import ObjectWrapper


def setup_logging(loglevel="INFO", logfile=None):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    kwargs = {'level': numeric_level, 'format': '%(asctime)s.%(msecs)06d %(levelname)-8s %(message)s',
              'datefmt': '%Y-%m-%d, %H:%M:%S', 'filemode': 'w'}
    if logfile:
        kwargs['filename'] = logfile
    logging.basicConfig(**kwargs)
    #https://stackoverflow.com/a/26433913/5122790
    warnings.formatwarning = lambda message, category, filename, lineno, file=None, line=None: f"{filename.replace(abspath(join(dirname(__file__), '..'))+sep, '')}:{lineno}: {category.__name__}: {message}\n"


#TODO make compatible with logging
class CustomIO(ObjectWrapper):
    DEFAULT_DATE_FORMAT = '%Y-%m-%d, %H:%M:%S.%f'

    def __init__(self, overwrites_stream, add_date=True, date_format=None, ctx=None):
        super(CustomIO, self).__init__(overwrites_stream)
        self.overwrites_stream = overwrites_stream
        self.ctx = ctx
        self.add_date = add_date
        self.initialized = False
        self.saver_io = io.StringIO()
        self.date_format = date_format or self.DEFAULT_DATE_FORMAT

    def write(self, *args, **kwargs):
        bkp_args = args #why the breakpoint? -> a if I print the pandas-dataframe the first line is missing (maybe just .replace("\n", "\r\n") or something on the string?!)
        if self.add_date:
            if not self.initialized:
                self.initialized = True
                args = tuple([datetime.now().strftime(self.date_format)+" "+args[0]] + list(args[1:]))
            elif args[0].endswith("\n"):
                args = tuple([args[0]+datetime.now().strftime(self.date_format)+" "]+list(args[1:]))
            if "\n" in args[0][1:-1]:
                args = tuple(["\n".join([args[0].split("\n")[0]]+[datetime.now().strftime(self.date_format)+i for i in args[0].split("\n")[1:]])]+list(args[1:]))
        self.saver_io.write(*args, **kwargs)
        return self._wrapped.write(*bkp_args, **kwargs)

    @staticmethod
    def init(ctx):
        sys.stdout = CustomIO(sys.stdout, ctx=ctx)
        sys.stderr = CustomIO(sys.stderr, ctx=ctx)
        return sys.stdout, sys.stderr

    @staticmethod
    @contextlib.contextmanager
    def context():
        yield CustomIO.init(None)
        sys.stderr = sys.stderr.overwrites_stream
        sys.stdout = sys.stdout.overwrites_stream

    def getvalue(self) -> str:
        lines = self.saver_io.getvalue().split("\n")
        lines = [((l.split("\r")[0] if self.add_date else "")+l.split("\r")[-1]) if "\r" in l else l for l in lines]
        return "\n".join(lines[:-1])

    def __eq__(self, other):
        return self._wrapped == getattr(other, '_wrapped', other)


if __name__ == "__main__":
    from tqdm import tqdm
    from time import sleep

    with CustomIO.context() as (new_out, new_err):
        print("HALLO", "wtf")
        print("next one")
        for i in tqdm(range(10)):
            sleep(0.1)

    print(new_out.getvalue())
    print(new_err.getvalue())