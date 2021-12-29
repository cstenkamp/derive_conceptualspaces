import logging
import contextlib
import io, sys
from datetime import datetime



def setup_logging(loglevel=None, logfile=None):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    kwargs = {'level': numeric_level, 'format': '%(asctime)s.%(msecs)06d %(levelname)-8s %(message)s',
              'datefmt': '%Y-%m-%d, %H:%M:%S', 'filemode': 'w'}
    if logfile:
        kwargs['filename'] = logfile
    logging.basicConfig(**kwargs)



class ObjectWrapper(object):
    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        return setattr(self._wrapped, name, value)

    def wrapper_getattr(self, name):
        """Actual `self.getattr` rather than self._wrapped.getattr"""
        try:
            return object.__getattr__(self, name)
        except AttributeError:  # py2
            return getattr(self, name)

    def wrapper_setattr(self, name, value):
        """Actual `self.setattr` rather than self._wrapped.setattr"""
        return object.__setattr__(self, name, value)

    def __init__(self, wrapped):
        """
        Thin wrapper around a given object
        """
        self.wrapper_setattr('_wrapped', wrapped)


#TODO make compatible with logging
class CustomIO(ObjectWrapper):
    DEFAULT_DATE_FORMAT = '%Y-%m-%d, %H:%M:%S.%f'

    def __init__(self, overwrites_stream, add_date=True, date_format=None):
        super(CustomIO, self).__init__(overwrites_stream)
        self.overwrites_stream = overwrites_stream
        self.add_date = add_date
        self.initialized = False
        self.saver_io = io.StringIO()
        self.date_format = date_format or self.DEFAULT_DATE_FORMAT

    def write(self, *args, **kwargs):
        bkp_args = args
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
    def init():
        sys.stdout = CustomIO(sys.stdout)
        sys.stderr = CustomIO(sys.stderr)
        return sys.stdout, sys.stderr

    @staticmethod
    @contextlib.contextmanager
    def context():
        yield CustomIO.init()
        sys.stderr = sys.stderr.overwrites_stream
        sys.stdout = sys.stdout.overwrites_stream

    def getvalue(self) -> str:
        lines = self.saver_io.getvalue().split("\n")
        lines = [((l.split("\r")[0] if self.add_date else "")+l.split("\r")[-1]) if "\r" in l else l for l in lines]
        return "\n".join(lines[:-1])

    def __eq__(self, other):
        return self._wrapped == getattr(other, '_wrapped', other)