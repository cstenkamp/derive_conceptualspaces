import sys
import os
from os.path import splitext
from datetime import datetime, timedelta
from collections import deque
from itertools import islice
from threading import Thread
from time import sleep
from multiprocessing import Queue

from tqdm import tqdm

from derive_conceptualspace.util.jsonloadstore import DependencyError

class IsDoneException(Exception):
    pass

#see https://stackoverflow.com/a/13733227/5122790
class SkipContext:
    def __enter__(self):
        self.args = None
        return self
    def __exit__(self, type, value, tb):
        if type is IsDoneException:
            self.args = value.args[0]
            return True


class InterruptibleLoad():
    def __init__(self, ctx, name_with_ending, loader=None, metainf_countervarnames=None, metainf_ignorevarnames=None):
        self.ctx = ctx
        self.name_with_ending = name_with_ending
        self.loader = loader
        self.metainf_countervarnames = metainf_countervarnames or [] #these ones we sum with old ones, others besides this and INTERRUPTED_AT we will assert to be equal
        self.metainf_ignorevarnames = metainf_ignorevarnames or []

    def __enter__(self):
        basename, ext = splitext(self.name_with_ending)
        self.loader = self.loader or (lambda **kw: kw[basename])
        try:
            tmp = self.ctx.p.get_file_by_config("", basename, postfix="INTERRUPTED")
            tmp2, self.old_metainf = self.ctx.p.load(tmp, f"{basename}_CONTINUE", silent=True, required_metainf=["INTERRUPTED_AT"], return_metainf=True, loader=self.loader)
            if "NEWLY_INTERRUPTED" in self.old_metainf:
                del self.old_metainf["NEWLY_INTERRUPTED"]
            self.kwargs = dict(continue_from=(tmp2, self.old_metainf, self.metainf_countervarnames, self.metainf_ignorevarnames))
        except (FileNotFoundError, DependencyError):
            self.kwargs = {}
        return self

    def save(self, metainf, **kwargs):
        if self.kwargs.get("continue_from"):
            if self.old_metainf.get("INTERRUPTED_IN") != metainf.get("INTERRUPTED_IN"):
                self.old_metainf["INTERRUPTED_AT"] = 0
            metainf = {**metainf, **{k: v+self.old_metainf.get(k,0) for k, v in metainf.items() if k in self.metainf_countervarnames}}
            if metainf.get("NEWLY_INTERRUPTED"):
                assert metainf["INTERRUPTED_AT"] > self.old_metainf["INTERRUPTED_AT"], "Not a single Iteration?!"
            assert all(v == self.old_metainf[k] for k, v in metainf.items() if k not in self.metainf_countervarnames+self.metainf_ignorevarnames+["NEWLY_INTERRUPTED", "INTERRUPTED_AT", "KEYBORD_INTERRUPTED", "INTERRUPTED_IN"])
            metainf["N_RUNS"] = self.old_metainf.get("N_RUNS", 0) + 1
            overwrite_old = self.old_metainf
        else:
            if metainf.get("NEWLY_INTERRUPTED"):
                metainf["N_RUNS"] = 1
            overwrite_old = False
        self.ctx.p.save(self.name_with_ending, metainf=metainf, overwrite_old=overwrite_old, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

########################################################################################################################
########################################################################################################################

class Interruptible():
    def __init__(self, iterable, append_var, metainf_var, shutdown_time=70, timeavg_samples=10, continue_from=None, pgbar=None, total=None, exc=False, contains_mp=False, name=None):
        #pgbar is either None or a string. If it's not None, we will wrap in tqdm and use the var-value as desc
        self.iterable = iterable
        self.append_var = append_var
        self.shutdown_time = shutdown_time
        self.timeavg_samples = timeavg_samples
        self.continue_from = continue_from
        self.pgbar = pgbar
        self.metainf_var = metainf_var
        self.kb_int = False
        self.total = total
        self.exc = exc
        self.interrupted = False
        self.n = 0
        self.contains_mp = contains_mp
        self.name = name

    def __enter__(self):
        self.interrupt_time = None
        if os.getenv("KILLED_AT"):
            self.interrupt_time = datetime.strptime(os.environ["KILLED_AT"], "%d.%m.%Y, %H:%M:%S")-timedelta(seconds=self.shutdown_time)
            print(f"This loop will interrupt at {self.interrupt_time.strftime('%d.%m.%Y, %H:%M:%S')}")
        self.full_len = len(self.iterable) if not self.total else self.total
        if self.continue_from is not None:
            old_results, self.old_metainf, countervarnames, ignorevarnames = self.continue_from
            if self.old_metainf is not None: assert self.metainf_var is not None
            assert all(v == self.old_metainf[k] for k, v in self.metainf_var.items()), "The metainf changed!"
            self.append_olds(old_results, with_assert=(self.old_metainf.get("INTERRUPTED_IN") == self.name))
            if self.old_metainf.get("INTERRUPTED_IN") != self.name:
                if any(len(i)==self.full_len for i in self.append_var if i is not None): #TODO just "any" is bad
                    print(f"Loop for {self.pgbar or self.name} is done.")
                    raise IsDoneException(self.append_var)
            print(f"Continuing a previously interrupted loop ({self.old_metainf['N_RUNS']} runs already). Starting at element {self.old_metainf['INTERRUPTED_AT'] + 1}/{self.full_len}")
            try:
                self.iterable = self.iterable[self.old_metainf["INTERRUPTED_AT"]:]
            except TypeError:
                self.iterable = islice(self.iterable, self.old_metainf["INTERRUPTED_AT"], None)
            sys.stderr.flush(); sys.stdout.flush()
        else:
            self.old_metainf = None
        self.n_elems = len(self.iterable) if hasattr(self.iterable, "__len__") else (self.total - self.old_metainf.get('INTERRUPTED_AT', 0) if self.old_metainf else self.total)
        if self.contains_mp:
            if self.interrupt_time is None:
                self.comqu = None
            else:
                self.comqu = Queue()
                self.interrupt_thread = Thread(target=self._interrupt_mainthread, daemon=True)
                self.interrupt_thread.start()
        return self

    def _interrupt_mainthread(self):
        while datetime.now() < self.interrupt_time:
            sleep(1)
        self.comqu.put("kill")


    def append_olds(self, old_results, with_assert=True):
        if len(self.append_var) > 1:
            enum = old_results.values() if isinstance(old_results, dict) else old_results
            for n, part in enumerate(enum):
                if part is None or self.append_var[n] is None:
                    continue
                if isinstance(part, dict):
                    for k, v in part.items():
                        self.append_var[n][k] = v
                elif isinstance(part, list):
                    for elem in part:
                        self.append_var[n].append(elem)
                if with_assert:
                    assert len(self.append_var[n]) == self.old_metainf["INTERRUPTED_AT"]
            return max(len(i) for i in enum if i)
        else:
            for elem in old_results:
                self.append_var.append(elem)
            if with_assert:
                assert len(self.append_var) == self.old_metainf["INTERRUPTED_AT"]
            return len(old_results)

    def __iter__(self):
        self.last_times = deque([datetime.now()], maxlen=self.timeavg_samples)
        if self.pgbar:
            self.tqdm = tqdm(self.iterable, desc=self.pgbar, total=self.n_elems)
        self.iter = enumerate(self.iterable) if self.pgbar is None else enumerate(self.tqdm)
        return self

    def __next__(self):
        self.last_times.append(datetime.now())
        timedeltas = [self.last_times[i]-self.last_times[i-1] for i in range(1, len(self.last_times))]
        average_timedelta = sum(timedeltas, timedelta(0)) / len(timedeltas) # https://stackoverflow.com/a/3617540/5122790
        if (self.interrupt_time and datetime.now()+average_timedelta*0.9 >= self.interrupt_time):
            self.after_interrupt()
            raise StopIteration()
        self.n, elem = next(self.iter)
        return elem

    def after_interrupt(self, kb=False):
        if kb:
            self.metainf_var["KEYBORD_INTERRUPTED"] = True
        if self.continue_from is not None:
            print(f"Interrupted at iteration {self.n}/{self.n_elems} ({self.old_metainf['INTERRUPTED_AT']+self.n}/{self.full_len})"+("" if kb else " because we will hit the wall-time!"))
            self.metainf_var["INTERRUPTED_AT"] = self.n+self.old_metainf["INTERRUPTED_AT"]
        else:
            print(f"Interrupted at iteration {self.n}/{self.n_elems}"+("" if kb else " because we will hit the wall-time!"))
            self.metainf_var["INTERRUPTED_AT"] = self.n
        if self.name is not None:
            self.metainf_var["INTERRUPTED_IN"] = self.name
        self.metainf_var["NEWLY_INTERRUPTED"] = True
        if hasattr(self, "tqdm"):
            self.tqdm.close()
        self.interrupted = True
        if self.exc:
            raise InterruptedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, KeyboardInterrupt):
            self.after_interrupt(kb=True)
            return True
        if not self.interrupted:
            self.old_metainf["INTERRUPTED_AT"] = 0

    def notify(self, results, exception):
        self.n = self.append_olds(results, with_assert=False)
        if exception is not False:
            self.after_interrupt(kb=isinstance(exception, KeyboardInterrupt))
        return self.append_var

########################################################################################################################
########################################################################################################################


if os.getenv("WALL_SECS"):
    wall_secs = int(os.environ['WALL_SECS'])
    killed_at = datetime.now() + timedelta(seconds=wall_secs)
    os.environ["KILLED_AT"] = killed_at.strftime("%d.%m.%Y, %H:%M:%S")
    print(f"Walltime: {wall_secs}s. Will be killed at {os.environ['KILLED_AT']}")
