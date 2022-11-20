from typing import Callable, Iterable, Iterator


def mapf(funcs: Iterable[Callable], *args, **kwargs) -> Iterator:
    """
    Apply each function in iterable to *args and **kwargs
    """
    for f in funcs:
        yield f(*args, **kwargs)