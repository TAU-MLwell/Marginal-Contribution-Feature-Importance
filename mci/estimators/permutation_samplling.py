import numpy as np
from typing import Optional, Sequence
from pandas import DataFrame
from tqdm.auto import tqdm

from mci.estimators.base_estimator import BaseEstimator
from mci.estimators.contribution_tracker import ContributionTracker
from mci.evaluators import EvaluationFunction
from mci.mci_values import MciValues
from mci.utils.estimators_util import context_to_key
from mci.utils.type_hints import MultiVariateArray, UniVariateArray


class PermutationSampling(BaseEstimator):

    def __init__(self,
                 evaluator: EvaluationFunction,
                 n_permutations: int,
                 n_processes: int = 5,
                 chunk_size: int = 2**12,
                 max_context_size: int = 100000,
                 noise_confidence: float = 0.05,
                 noise_factor: float = 0.1,
                 track_all: bool = False):

        super(PermutationSampling, self).__init__(evaluator=evaluator,
                                                  n_processes=n_processes,
                                                  chunk_size=chunk_size,
                                                  max_context_size=max_context_size,
                                                  noise_confidence=noise_confidence,
                                                  noise_factor=noise_factor,
                                                  track_all=track_all)
        self._n_permutations = n_permutations

    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[Sequence[str]] = None) -> MciValues:
        if not isinstance(x, DataFrame):
            assert x is not None, "feature names must be provided if x is not a dataframe"
            x = DataFrame(x, columns=feature_names)

            if x_test is not None and not isinstance(x_test, DataFrame):
                x_test = DataFrame(x_test, columns=feature_names)

        feature_names = list(x.columns)
        permutations_sample = self._sample_permutations(feature_names)
        suffixes = [p[:i] for p in permutations_sample for i in range(len(p))]
        evaluations = self._multiprocess_eval_subsets(suffixes, x, y, x_test, y_test)

        tracker = ContributionTracker(n_features=len(feature_names), track_all=self._track_all)
        for p in tqdm(permutations_sample):
            for i in range(len(p)-1):
                suffix = p[:i]
                suffix_with_f = p[:i+1]
                contribution = evaluations[context_to_key(suffix_with_f)] - evaluations[context_to_key(suffix)]
                tracker.update_value(feature_idx=feature_names.index(p[i]),
                                     contribution=contribution,
                                     context=set(suffix))
        return MciValues.create_from_tracker(tracker, feature_names)

    def _sample_permutations(self, feature_names: Sequence[str]):
        np.random.seed(0)
        return [list(np.random.permutation(feature_names))
                for _ in range(self._n_permutations)]
