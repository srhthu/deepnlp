"""
Toolkit to parallel execute a list of tasks based on multi-processing.
"""
from multiprocessing import Pool, Process, Queue, JoinableQueue
import os, time, random
from tqdm import tqdm
import queue
import numpy as np

def mp_map(foo, tasks, n, chunksize = 1):
    pool = Pool(n)
    res = pool.map(foo, tasks, chunksize=chunksize)
    pool.close()
    pool.join()
    return res

def mp_map_async(foo, tasks, n, chunksize=1):
    pool = Pool(n)
    res = pool.map(foo, tasks, chunksize = chunksize)
    pool.close()
    pool.join()
    return res

def mp_apply_async(foo, tasks, n):
    pool = Pool(n)
    res = []
    for task in tasks:
        res.append(pool.apply_async(func = foo, args = (task,)))
    pool.close()
    pool.join()
    res = [r.get(timeout = 1) for r in res]
    return res

# Warning: this function is very slow
def mp_apply_queue(foo, tasks, n):
    q = JoinableQueue(100)
    qr = JoinableQueue(100)

    def wrapped_foo(q: Queue, qr: Queue):
        while True:
            try:
                w_data = q.get(True, 0.1) # no block
            except queue.Empty as e:
                break
            rtn_data = foo(w_data[1])
            qr.put((w_data[0], rtn_data))
        return None

    # Process to put data into queue
    def feed_f():
        for i, d in enumerate(tasks):
            q.put((i, d))
    feed_p = Process(target = feed_f)
    feed_p.start()
    
    # Start sub processes
    pro_l = [Process(target = wrapped_foo, args = (q, qr)) for _ in range(n)]
    for p in pro_l:
        p.start()

    # Collect results from return queue
    res = []
    bar = tqdm(total = len(tasks), ncols = 80)
    while True:
        if len(res) == len(tasks):
            break
        res.append(qr.get())
        bar.update()
    
    # join all processes
    feed_p.join()
    for p in pro_l:
        p.join()

    # rank tasks
    res.sort(key = lambda k: k[0])
    return [k[1] for k in res]

def mp_chunk_apply(foo, tasks, n):
    chunksize, mod = divmod(len(tasks), n)
    pro_l = []
    q = Queue()

    wrap_tasks = [(i, t) for i,t in enumerate(tasks)]
    def batch_foo(b_tasks, q:Queue):
        rl = [(t[0], foo(t[1])) for t in b_tasks]
        q.put(rl)
    
    num_tk = np.ones(n, dtype = np.int64) * chunksize
    for i in range(mod):
        num_tk[i] += 1
    endp = [0] + np.cumsum(num_tk).tolist()

    for i in range(n):
        pro_l.append(Process(target = batch_foo, 
                              args = (wrap_tasks[endp[i]: endp[i+1]], q)))
    for p in pro_l:
        p.start()
    
    res = []
    count = 0
    while True:
        if count >= n:
            break
        res.extend(q.get())
        count += 1
    for p in pro_l:
        p.join()
    # rank tasks
    res.sort(key = lambda k: k[0])
    return [k[1] for k in res]
    


if __name__ == '__main__':
    def foo(k):
        # print(f'{os.getpid()}, {k}')
        #time.sleep(random.random() * 3)
        time.sleep(1)
        return {'data': k}

    # res = mp_map_async(foo, range(12), 3)
    # res = mp_apply_async(foo, range(12), 3)
    # res = mp_apply_queue(foo, range(12), 3)
    res = mp_chunk_apply(foo, range(12), 3)
    print(res)