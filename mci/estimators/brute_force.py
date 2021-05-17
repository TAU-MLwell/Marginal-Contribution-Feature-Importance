from itertools import combinations
from typing import Optional, Tuple, List, Dict
from pandas import DataFrame
from mci.estimators.contribution_tracker import ContributionTracker
from mci.utils.pac_noise import pac_noise
from mci.utils.type_hints import MultiVariateArray, UniVariateArray
from mci.estimators.base_estimator import BaseEstimator
from mci.mci_values import MciValues
from mci.utils.estimators_util import context_to_key


class BruteForce(BaseEstimator):

    """Exhaustive search over all features subsets (optimal algorithm)"""

    def mci_values(self,
                   x: MultiVariateArray,
                   y: UniVariateArray,
                   x_test: Optional[MultiVariateArray] = None,
                   y_test: Optional[UniVariateArray] = None,
                   feature_names: Optional[List[str]] = None) -> MciValues:
        if not isinstance(x, DataFrame):
            assert x is not None, "feature names must be provided if x is not a dataframe"
            x = DataFrame(x, columns=feature_names)

            if x_test is not None and not isinstance(x_test, DataFrame):
                x_test = DataFrame(x_test, columns=feature_names)

        feature_names = list(x.columns)

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
                              y_test: Optional[UniVariateArray]) -> Dict[Tuple[str, ...], float]:
        feature_names = list(x.columns)
        n_features = len(feature_names)

        sizes = range(min(n_features, self._max_context_size + 1) + 1)
        all_subsets = [c for size in sizes for c in combinations(feature_names, size)]
        return self._multiprocess_eval_subsets(all_subsets, x, y, x_test, y_test)
