import time
from multiprocessing import JoinableQueue, Process

from tqdm import tqdm

class WorkerPool():
    def __init__(self, n_workers, workerobj, pgbar=None, comqu=None):
        self.qu = JoinableQueue()
        self.donequ = JoinableQueue()
        self.workers = [Worker(self.qu, self.donequ, workerobj, num) for num in range(n_workers)]
        self.pgbar = pgbar
        self.known_deaths = []
        self.comqu = comqu

    def __enter__(self):
        return self

    def work(self, iterable, func):
        iterable = list(enumerate(iterable))
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
                    while len(results) < len(iterable):
                        self.bookkeep()
                        while not self.donequ.empty():
                            results.append(self.donequ.get())
                            pgbar.update(1)
                        if self.comqu is not None and not self.comqu.empty():
                            raise InterruptedError()
            else:
                while len(results) < len(iterable):
                    self.bookkeep()
                    time.sleep(0.05)
                    if self.comqu is not None and not self.comqu.empty():
                        raise InterruptedError()
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
    def __init__(self, queue, donequ, obj, num):
        Process.__init__(self)
        self.queue = queue
        self.donequ = donequ
        self.obj = obj
        self.num = num
        self.func = None

    def run(self):
        while True:
            try:
                self.item = self.queue.get()
                self.donequ.put((self.item[0], self.func(self.obj, self.item[1])))
                self.queue.task_done()
            except KeyboardInterrupt:
                break
