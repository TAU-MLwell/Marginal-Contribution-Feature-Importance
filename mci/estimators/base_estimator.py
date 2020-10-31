from abc import  ABC, abstractmethod
from typing import Optional, Sequence
from mci.evaluators import MultiVariateArray, UniVariateArray
from mci.mci_values import MciValues


class BaseEstimator(ABC):

    @abstractmethod
    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[Sequence[str]] = None) -> MciValues:

        raise NotImplementedError()
