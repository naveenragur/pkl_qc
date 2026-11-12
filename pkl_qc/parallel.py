"""Process-per-task parallel map that fails loudly instead of hanging if a worker dies
(e.g. killed by the login-node CPU-time limit)."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


def run_tasks(func, jobs, nworkers, initializer=None, initargs=(), on_error=None):
    """Yield func(job) results in completion order; each task runs in a fresh process.
    If a worker dies, on_error(job, exc) supplies the result (default: re-raise)."""
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=max(1, min(nworkers, len(jobs))), max_tasks_per_child=1,
                             initializer=initializer, initargs=initargs) as ex:
        futs = {ex.submit(func, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                yield f.result()
            except BrokenProcessPool as e:
                if on_error is None:
                    raise
                yield on_error(futs[f], e)
