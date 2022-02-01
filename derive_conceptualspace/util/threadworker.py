import time
from multiprocessing import JoinableQueue, Process

from tqdm import tqdm

class WorkerPool():
    def __init__(self, n_workers, workerobj, pgbar=None):
        self.qu = JoinableQueue()
        self.donequ = JoinableQueue()
        self.workers = [Worker(self.qu, self.donequ, workerobj, num) for num in range(n_workers)]
        self.pgbar = pgbar

    def __enter__(self):
        return self

    def work(self, iterable, func):
        for elem in iterable:
            self.qu.put(elem)
        for worker in self.workers:
            worker.func = func
            worker.start()
        results = []
        if self.pgbar:
            with tqdm(total=len(iterable), desc=self.pgbar) as pgbar:
                while len(results) < len(iterable):
                    while not self.donequ.empty():
                        results.append(self.donequ.get())
                        pgbar.update(1)
        else:
            while len(results) < len(iterable):
                time.sleep(0.05)
        return [i[1] for i in sorted(results, key=lambda x: iterable.index(x[0]))]

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


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
                item = self.queue.get()
                self.donequ.put((item, self.func(self.obj, item)))
                self.queue.task_done()
            except KeyboardInterrupt:
                break
