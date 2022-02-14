import sys
import os
from os.path import splitext
from datetime import datetime, timedelta
from collections import deque
from itertools import islice

from tqdm import tqdm

from derive_conceptualspace.util.jsonloadstore import DependencyError


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
            metainf = {**metainf, **{k: v+self.old_metainf.get(k,0) for k, v in metainf.items() if k in self.metainf_countervarnames}}
            if metainf.get("NEWLY_INTERRUPTED"):
                assert metainf["INTERRUPTED_AT"] > self.old_metainf["INTERRUPTED_AT"], "Not a single Iteration?!"
            assert all(v == self.old_metainf[k] for k, v in metainf.items() if k not in self.metainf_countervarnames+self.metainf_ignorevarnames+["NEWLY_INTERRUPTED", "INTERRUPTED_AT", "KEYBORD_INTERRUPTED"])
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
    def __init__(self, iterable, append_var, metainf_var, shutdown_time=70, timeavg_samples=10, continue_from=None, pgbar=None, total=None, exc=False):
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

    def __enter__(self):
        return self

    def append_olds(self, old_results):
        if len(self.append_var) > 1:
            for n, part in enumerate(old_results.values()):
                if isinstance(part, dict):
                    for k, v in part.items():
                        self.append_var[n][k] = v
                assert len(self.append_var[n]) == self.old_metainf["INTERRUPTED_AT"]
        else:
            for elem in old_results:
                self.append_var.append(elem)
            assert len(self.append_var) == self.old_metainf["INTERRUPTED_AT"]

    def __iter__(self):
        self.interrupt_time = None
        if os.getenv("KILLED_AT"):
            self.interrupt_time = datetime.strptime(os.environ["KILLED_AT"], "%d.%m.%Y, %H:%M:%S")-timedelta(seconds=self.shutdown_time)
            print(f"This loop will interrupt at {self.interrupt_time.strftime('%d.%m.%Y, %H:%M:%S')}")
        self.last_times = deque([datetime.now()], maxlen=self.timeavg_samples)
        self.full_len = len(self.iterable) if not self.total else self.total
        if self.continue_from is not None:
            old_results, self.old_metainf, countervarnames, ignorevarnames = self.continue_from
            if self.old_metainf is not None: assert self.metainf_var is not None
            print(f"Continuing a previously interrupted loop ({self.old_metainf['N_RUNS']} runs already). Starting at element {self.old_metainf['INTERRUPTED_AT'] + 1}/{self.full_len}")
            assert all(v == self.old_metainf[k] for k, v in self.metainf_var.items()), "The metainf changed!"
            self.append_olds(old_results)
            try:
                self.iterable = self.iterable[self.old_metainf["INTERRUPTED_AT"]:]
            except TypeError:
                self.iterable = islice(self.iterable, self.old_metainf["INTERRUPTED_AT"], None)
            sys.stderr.flush(); sys.stdout.flush()
        else:
            self.old_metainf = None
        self.n_elems = len(self.iterable) if hasattr(self.iterable, "__len__") else (self.total - self.old_metainf.get('INTERRUPTED_AT', 0) if self.old_metainf else self.total)
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
        if self.continue_from is not None:
            print(f"Interrupted at iteration {self.n}/{self.n_elems} ({self.old_metainf['INTERRUPTED_AT']+self.n}/{self.full_len})"+("" if kb else "because we will hit the wall-time!"))
            self.metainf_var["INTERRUPTED_AT"] = self.n+self.old_metainf["INTERRUPTED_AT"]
        else:
            print(f"Interrupted at iteration {self.n}/{self.n_elems}"+("" if kb else "because we will hit the wall-time!"))
            self.metainf_var["INTERRUPTED_AT"] = self.n
        self.metainf_var["NEWLY_INTERRUPTED"] = True
        if hasattr(self, "tqdm"):
            self.tqdm.close()
        self.interrupted = True
        if self.exc:
            raise InterruptedError()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, KeyboardInterrupt):
            self.metainf_var["KEYBORD_INTERRUPTED"] = True
            self.after_interrupt(kb=True)
            return True

########################################################################################################################
########################################################################################################################


if os.getenv("WALL_SECS"):
    wall_secs = int(os.environ['WALL_SECS'])
    killed_at = datetime.now() + timedelta(seconds=wall_secs)
    os.environ["KILLED_AT"] = killed_at.strftime("%d.%m.%Y, %H:%M:%S")
    print(f"Walltime: {wall_secs}s. Will be killed at {os.environ['KILLED_AT']}")
