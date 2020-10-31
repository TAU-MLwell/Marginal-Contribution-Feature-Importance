from itertools import combinations
from typing import Optional, Iterable, Tuple, Set, List, Dict
from pandas import DataFrame
from tqdm.auto import tqdm

from mci.estimators.contribution_tracker import ContributionTracker
from mci.utils.pac_noise import pac_noise
from mci.evaluators import EvaluationFunction, MultiVariateArray, UniVariateArray
from mci.estimators.base_estimator import BaseEstimator
from mci.mci_values import MciValues
from mci.utils.estimators_util import context_to_key
from mci.utils.multi_process_lst import multi_process_lst


class BruteForce(BaseEstimator):

    """Exhaustive search over all features subsets (optimal algorithm)"""

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

    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[List[str]] = None) -> MciValues:
        if not isinstance(x, DataFrame):
            assert x is not None, "feature names must be provided if x is not a dataframe"
            x = DataFrame(x, columns=feature_names)

            if x_test is not None and  not isinstance(x_test, DataFrame):
                x_test = DataFrame(x_test, columns=feature_names)

        feature_names = x.columns

        subsets_evals = self._evaluate_all_subsets(x, y, x_test, y_test)
        tracker = ContributionTracker(n_features=len(feature_names), track_all=self._track_all)
        for f_idx, f in enumerate(feature_names):
            context_features = set(feature_names) - {f}
            for context_size in range(min(len(feature_names), self._max_context_size) + 1):
                noise = pac_noise(len(x), len(feature_names), context_size, self._noise_confidence, self._noise_factor)
                for context in combinations(context_features, context_size):
                    contribution = subsets_evals[context_to_key(set(context).union({f}))] -\
                                   subsets_evals[context_to_key(context)]
                    tracker.update_value(feature_idx=f_idx, contribution=contribution,
                                         context=set(context), noise_tolerance=noise)
        return MciValues.create_from_tracker(tracker, feature_names)

    def _evaluate_all_subsets(self,
                              x: DataFrame,
                              y: UniVariateArray,
                              x_test: Optional[DataFrame],
                              y_test: Optional[UniVariateArray]) -> Dict[Tuple[str], float]:
        feature_names = x.columns
        n_features = len(feature_names)
        evaluations = {}

        sizes = range(min(n_features, self._max_context_size + 1) + 1)
        subsets = [c for size in sizes for c in combinations(feature_names, size)]
        pbar = tqdm(total=len(subsets))
        for eval_results in multi_process_lst(lst=subsets, apply_on_chunk=self._evaluate_subsets_chunk,
                                              chunk_size=self._chunk_size, n_processes=self._n_processes,
                                              args=(x, y, x_test, y_test)):
            evaluations.update(eval_results)
            pbar.update(len(eval_results))
        return evaluations

    def _evaluate_subsets_chunk(self,
                                contexts: Iterable[Set[str]],
                                x: DataFrame,
                                y: UniVariateArray,
                                x_test: Optional[DataFrame],
                                y_test: Optional[UniVariateArray]) -> Dict[Tuple[str], float]:
        evaluations = {}
        for c in contexts:
            if x_test is not None:
                x_test = x_test[list(c)]
            evaluations[context_to_key(c)] = self._evaluator(x[list(c)], y, x_test, y_test)
        return evaluations
