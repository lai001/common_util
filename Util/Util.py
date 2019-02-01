from multiprocessing import cpu_count

import threadpool


def creat_thread(num_workers: int = cpu_count(), callable_: function = None, args_list: list = None,
                 callback: function = None, completion_handler: function = None):
    if callable_ is None or args_list is None or callback is None:
        raise ValueError("不能为空")
    pool = threadpool.ThreadPool(num_workers)
    requests = threadpool.makeRequests(callable_, args_list, callback)
    [pool.putRequest(req) for req in requests]
    pool.wait()
    if completion_handler is not None:
        completion_handler()
