from functools import wraps
from typing import List, Tuple, Union


def apply_at(pos: Union[List[int], Tuple[int], int]):
    pos = [pos] if isinstance(pos, int) else pos

    def decorator(func):
        @wraps(func)
        def wrapper(*xs, **kwargs):
            pass
            # TODO
