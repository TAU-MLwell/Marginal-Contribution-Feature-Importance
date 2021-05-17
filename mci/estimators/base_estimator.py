from abc import ABC, abstractmethod
from typing import Optional, Sequence, List, Iterable, Dict, Tuple
from tqdm.auto import tqdm
from pandas import DataFrame

from mci.evaluators.evaluator_function import MultiVariateArray, UniVariateArray, EvaluationFunction
from mci.mci_values import MciValues
from mci.utils.estimators_util import context_to_key
from mci.utils.multi_process_lst import multi_process_lst


class BaseEstimator(ABC):

    def __init__(self,
                 evaluator: EvaluationFunction,
                 n_processes: int = 5,
                 chunk_size: int = 20,
                 max_context_size: int = 100000,
                 noise_confidence: float = 0.05,
                 noise_factor: float = 0.1,
                 track_all: bool = False):
        """
        :param evaluator: features subsets evaluation function
        :param n_processes: number of process to use
        :param chunk_size: max number of subsets to evaluate at each process at a time
        :param max_context_size: max feature subset size to evaluate as feature context
        :param noise_confidence: PAC learning error bound confidence (usually noted as delta for PAC)
        :param noise_factor: a scalar to multiple by the PAC learning error bound
        :param track_all: a bool indicates whether to save all observed contributions and not just max
        """

        self._evaluator = evaluator
        self._n_processes = n_processes
        self._chunk_size = chunk_size
        self._max_context_size = max_context_size
        self._noise_factor = noise_factor
        self._noise_confidence = noise_confidence
        self._track_all = track_all

    @abstractmethod
    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[Sequence[str]] = None) -> MciValues:

        raise NotImplementedError()

    def _multiprocess_eval_subsets(self,
                                   subsets: List[Iterable[str]],
                                   x: DataFrame,
                                   y: UniVariateArray,
                                   x_test: Optional[DataFrame],
                                   y_test: Optional[UniVariateArray] = None) -> Dict[Tuple[str, ...], float]:
        subsets = list(set(context_to_key(c) for c in subsets))  # remove duplications

        evaluations: Dict[Tuple[str, ...], float] = {}
        pbar = tqdm(total=len(subsets))
        for eval_results in multi_process_lst(lst=subsets, apply_on_chunk=self._evaluate_subsets_chunk,
                                              chunk_size=self._chunk_size, n_processes=self._n_processes,
                                              args=(x, y, x_test, y_test)):
            evaluations.update(eval_results)
            pbar.update(len(eval_results))
        return evaluations

    def _evaluate_subsets_chunk(self,
                                subsets: List[Iterable[str]],
                                x: DataFrame,
                                y: UniVariateArray,
                                x_test: Optional[DataFrame],
                                y_test: Optional[UniVariateArray]) -> Dict[Tuple[str], float]:
        evaluations: Dict[Tuple[str, ...], float] = {}
        for s in subsets:
            evaluations[context_to_key(s)] = self._evaluator(x[list(s)], y, x_test[list(s)] if x_test is not None
            else None, y_test)
        return evaluations
