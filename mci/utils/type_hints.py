from typing import Union
from pandas import DataFrame, Series
from numpy import ndarray

"""type hints shortcut definitions"""

MultiVariateArray = Union[DataFrame, Series, ndarray]
UniVariateArray = Union[Series, ndarray, list]
