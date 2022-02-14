import queue
import threading
import os
import time
import signal
import functools

# https://gist.github.com/JevinJ/b7163af961acf92f50b6d0d6efb39daf

class Worker(threading.Thread):
    def __init__(self, tasks, results):
        super().__init__()
        self.tasks = tasks
        self.shutdown_flag = threading.Event()
        self.daemon = True
        self.results = results
        self.star = False

    def run(self):
        while not self.shutdown_flag.is_set():
            try:
                task = self.tasks.get_nowait()
                if self.star:
                    args = task[1][0]
                    if any(isinstance(i, str) and i.startswith("next_") for i in args):
                        elems = next(self.draw_from)
                        args = [i if not(isinstance(i, str) and i.startswith("next_")) else elems[int(i.split("_")[1])] for i in args]
                    self.results.append(task[0](*args))
                else:
                    self.results.append(task())
            except queue.Empty:
                break
            else:
                self.tasks.task_done()


class ThreadPool:
    '''
    Interruptable thread pool with multiprocessing.map-like function, KeyboardInterrupt stops the pool,
    KeyboardInterrupt can be caught and the pool can be continued.
    '''
    def __init__(self, n_procs=None, comqu=None):
        signal.signal(signal.SIGINT, self.interrupt_event)
        signal.signal(signal.SIGTERM, self.interrupt_event)
        self.comqu = comqu
        self.tasks = queue.Queue()
        self.threads = []
        self.results = []
        self.n_procs = n_procs or os.cpu_count()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def interrupt_event(self, signum, stack):
        self.stop()
        raise KeyboardInterrupt

    def start(self, star=False):
        for thread in self.threads:
            thread.draw_from = self.draw_from if hasattr(self, "draw_from") else None
            thread.star = star
            thread.start()

    def stop(self):
        for thread in self.threads:
            thread.shutdown_flag.set()
        for thread in self.threads:
            thread.join()

    def add_to_queue(self, func, *args, star=False):
        self.tasks.put(functools.partial(func, *args) if not star else (func, args))

    def map(self, func, iterable, star=False):
        for task in iterable:
            self.add_to_queue(func, task, star=star)
        self.reset()
        self.start(star=star)
        try:
            while self.isRunning() and not self.tasks.empty():
                if self.comqu is not None and not self.comqu.empty():
                    raise InterruptedError()
                time.sleep(.01)
        except (KeyboardInterrupt, InterruptedError) as e:
            self.stop()
            return self.results, e
        self.stop()
        return self.results, False

    def starmap(self, func, iterables, draw_from=None):
        self.draw_from = draw_from
        iterable = [i for i in iterables]
        return self.map(func, iterable, star=True)

    def reset(self):
        self.result = []
        self.threads = [Worker(self.tasks, self.results) for t in range(self.n_procs)]

    def isRunning(self):
        return any(thread.is_alive() for thread in self.threads)


#Return (num, True) if no exception(success), or (num, False) so we can retry.
def fetch(task_num):
    try:
        res = task_num*2
    except:
        return (task_num, None, False)
    return (task_num, res, True)

# #Demonstrating failures
# def run(task_num):
#     if random.randint(1, 10) < 5:
#         print(f'failed: {task_num}')
#         raise ValueError
#     print(f'OK: {task_num}')
#     time.sleep(1)


if __name__ == '__main__':
    nums = [i for i in range(100000)]
    #You can catch KeyboardInterrupt here and continue or pass and the program will exit.
    with ThreadPool() as pool:
        res = [result for result in pool.map(fetch, nums)]
    print()
    print("done")