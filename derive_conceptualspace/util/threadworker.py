import time
from multiprocessing import JoinableQueue, Process

from tqdm import tqdm

class WorkerPool():
    def __init__(self, n_workers, workerobj=None, pgbar=None, comqu=None):
        self.qu = JoinableQueue()
        self.prioqu = JoinableQueue()
        self.donequ = JoinableQueue()
        self.workers = [Worker(self.qu, self.prioqu, self.donequ, num, workerobj) for num in range(n_workers)]
        self.pgbar = pgbar if pgbar is None else pgbar+f" [{n_workers} procs]"
        self.known_deaths = []
        self.comqu = comqu

    def __enter__(self):
        return self

    def work(self, iterable, func, enum_start=0):
        iterable = list(enumerate(iterable, start=enum_start))
        self.iterable = iterable
        for elem in iterable:
            self.qu.put(elem)
        for worker in self.workers:
            worker.func = func
            worker.start()
        results = []
        try:
            if self.pgbar:
                with tqdm(total=len(iterable), desc=self.pgbar) as pgbar:
                    last_dead_workers = set()
                    died_at = -1
                    while len(results) < len(iterable):

                        dead_workers = set(i.name for i in self.workers if i.exitcode is not None)
                        if len(dead_workers) == len(self.workers):
                            raise InterruptedError()
                        if dead_workers > last_dead_workers:
                            died_at = pgbar.n
                            last_dead_workers = dead_workers
                        if died_at >= 0 and pgbar.n > died_at + (len(self.workers)-len(dead_workers))*2:
                            undone_elems = set(range(pgbar.n-died_at))-set(i[0] for i in results)
                            for elem in undone_elems:
                                self.prioqu.put((elem, dict(iterable)[elem]))
                            died_at = -1

                        self.bookkeep()
                        while not self.donequ.empty():
                            results.append(self.donequ.get())
                            pgbar.update(1)
                        if self.comqu is not None and not self.comqu.empty():
                            raise InterruptedError()
                        time.sleep(0.1)
            else:
                raise NotImplementedError("TODO - Do this if needed")
                # while len(results) < len(iterable):
                #     self.bookkeep()
                #     time.sleep(0.05)
                #     if self.comqu is not None and not self.comqu.empty():
                #         raise InterruptedError()
        except (KeyboardInterrupt, InterruptedError) as e:
            while not self.qu.empty():
                self.qu.get()
            time.sleep(5) #TODO make this a variable
            for worker in self.workers:
                worker.kill()
            while not self.donequ.empty():
                results.append(self.donequ.get())
                pgbar.update(1)
            missing_keys = set(range(max(i[0] for i in results))) - set(i[0] for i in results) #not all processes are equally fast, so some may be not done
            done_until = min(len(results), max(i[0] for i in results), *missing_keys)
            return [i[1] for i in sorted(results)[:done_until]], e
        return [i[1] for i in sorted(results)], False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for worker in self.workers:
            worker.kill()
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt): #TODO also Interrupted
            print()

    def bookkeep(self):
        for worker in self.workers:
            if worker.exitcode is not None and worker.name not in self.known_deaths:
                self.known_deaths.append(worker.name)
                print(f"{worker.name} died!")
                #TODO könnte gucken ob [i for i in self.iterable if i not in [i[0] for i in results]] aber whatever


class Worker(Process):
    def __init__(self, queue, prioqu, donequ, num, obj=None):
        Process.__init__(self)
        self.queue = queue
        self.prioqu = prioqu
        self.donequ = donequ
        self.obj = obj
        self.num = num
        self.func = None

    def run(self):
        while True:
            try:
                if not self.prioqu.empty():
                    self.item = self.prioqu.get()
                    print(f"A dropped job (number {self.item[0]}) was caught up")
                else:
                    self.item = self.queue.get()
                if self.obj is not None:
                    self.donequ.put((self.item[0], self.func(self.obj, self.item[1])))
                else:
                    self.donequ.put((self.item[0], self.func(self.item[1])))
                self.queue.task_done()
            except KeyboardInterrupt:
                break
