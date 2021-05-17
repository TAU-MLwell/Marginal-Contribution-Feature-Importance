from typing import Callable, Optional
from mci.utils.type_hints import MultiVariateArray, UniVariateArray


"""defines evaluation function type"""

EvaluationFunction = Callable[[MultiVariateArray, UniVariateArray,
                               Optional[MultiVariateArray], Optional[UniVariateArray]], float]

