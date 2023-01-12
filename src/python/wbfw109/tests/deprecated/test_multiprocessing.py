from multiprocessing import Pool, Process, Queue, TimeoutError
import os
import time


def f(x):
    return x * x


def f2(name: str, my_queue: Queue):
    my_queue.put([42, None, "hello from queue"])
    info("function f2")
    print("hello", name)


def info(title):
    print(title)
    print("module name:", __name__)
    print("parent process:", os.getppid())
    print("process id:", os.getpid())


if __name__ == "__main__":
    my_queue = Queue()
    p = Process(target=f2, args=("bob", my_queue))
    p.start()
    print(my_queue.get())
    p.join()

    print()
    # start 4 worker processes
    with Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20,))  # runs in *only* one process
        print(res.get(timeout=1))  # prints "400"

        # evaluate "os.getpid()" asynchronously
        res = pool.apply_async(os.getpid, ())  # runs in *only* one process
        print(res.get(timeout=1))  # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([f"getpid: {res.get(timeout=1)}" for res in multiple_results])

        # make a single worker sleep for 10 secs
        res = pool.apply_async(time.sleep, (10,))
        try:
            print(res.get(timeout=1))
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more work")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")

    # start methods in Unix are fork as default. Available on Unix only.
