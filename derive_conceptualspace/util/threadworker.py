import time
from multiprocessing import JoinableQueue, Process

from tqdm import tqdm

class WorkerPool():
    def __init__(self, n_workers, workerobj, pgbar=None):
        self.qu = JoinableQueue()
        self.donequ = JoinableQueue()
        self.workers = [Worker(self.qu, self.donequ, workerobj, num) for num in range(n_workers)]
        self.pgbar = pgbar
        self.known_deaths = []

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
        if self.pgbar:
            with tqdm(total=len(iterable), desc=self.pgbar) as pgbar:
                while len(results) < len(iterable):
                    self.bookkeep()
                    while not self.donequ.empty():
                        results.append(self.donequ.get())
                        pgbar.update(1)
        else:
            while len(results) < len(iterable):
                self.bookkeep()
                time.sleep(0.05)
        return [i[1] for i in sorted(results)]

    def __exit__(self, exc_type, exc_val, exc_tb):
        for worker in self.workers:
            worker.kill()

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
