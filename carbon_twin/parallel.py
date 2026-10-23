from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

log = logging.getLogger(__name__)


def n_blocks(n_cells: int, workers: int, n_rows: int, cap: int = 64) -> int:
    if n_cells <= 0 or workers <= 1 or n_rows <= 1:
        return 1
    return max(1, min(cap, n_rows, -(-2 * workers // n_cells)))


def row_block(df: pd.DataFrame, task: dict) -> pd.DataFrame:
    n = task.get("n_blocks", 1)
    if n <= 1:
        return df
    b = task["block"]
    return df.iloc[len(df) * b // n: len(df) * (b + 1) // n]


def run_blocked(
    tasks: list[dict],
    worker: Callable[[dict], list],
    *,
    workers: int,
    n_rows: int,
    initializer: Callable | None = None,
    initargs: tuple = (),
) -> Iterator[tuple[dict, pd.DataFrame | Exception]]:
    nb = n_blocks(len(tasks), workers, n_rows)
    nw = max(1, min(workers, len(tasks) * nb))
    log.info("%d cells x %d vessel blocks on %d workers", len(tasks), nb, nw)
    parts: dict[int, list] = {i: [None] * nb for i in range(len(tasks))}
    with ProcessPoolExecutor(max_workers=nw, initializer=initializer,
                             initargs=initargs) as ex:
        fut = {ex.submit(worker, {**t, "block": b, "n_blocks": nb}): (i, b)
               for i, t in enumerate(tasks) for b in range(nb)}
        try:
            for f in as_completed(fut):
                i, b = fut.pop(f)
                if i not in parts:
                    continue
                try:
                    parts[i][b] = f.result() or []
                except Exception as exc:
                    del parts[i]
                    yield tasks[i], exc
                    continue
                if all(p is not None for p in parts[i]):
                    yield tasks[i], pd.DataFrame([r for p in parts.pop(i) for r in p])
        finally:
            for f in fut:
                f.cancel()
