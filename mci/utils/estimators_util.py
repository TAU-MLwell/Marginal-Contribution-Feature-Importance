from typing import Iterable, Tuple
from pandas import DataFrame, Series
from mci.utils.type_hints import MultiVariateArray


def context_to_key(context: Iterable[str]) -> Tuple[str]:
    return tuple(sorted(context))


def is_empty(array: MultiVariateArray):
    if isinstance(array, DataFrame) or isinstance(array, Series):
        return array.empty
    else:
        return len(array) == 0
