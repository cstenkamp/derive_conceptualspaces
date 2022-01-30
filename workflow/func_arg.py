import time
import os
import signal

class TimeOutException(Exception):
    pass
def handler(signum, frame):
    os.environ["KILL_ME"] = "1"
signal.signal(signal.SIGALRM, handler)
if os.getenv("WALL_SECS"):
    print(f"walltime: {int(os.environ['WALL_SECS'])}")
    signal.alarm(int(os.environ['WALL_SECS']))


def func():
    print("Loop starting..")
    for num in range(10):
        time.sleep(1)
        print(num)
        if os.getenv("KILL_ME"):
            print("killing..")
            break
    print("Loop ended")
    print(num)