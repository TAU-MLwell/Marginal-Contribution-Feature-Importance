from typing import Optional, Sequence, Set, List, Dict, Tuple
from pandas import DataFrame
from tqdm.auto import tqdm

from mci.estimators.base_estimator import BaseEstimator
from mci.evaluators.evaluator_function import EvaluationFunction
from mci.mci_values import MciValues
from mci.utils.pac_noise import pac_noise
from mci.utils.type_hints import MultiVariateArray, UniVariateArray
from mci.utils.estimators_util import context_to_key
from mci.estimators.contribution_tracker import ContributionTracker


class BeamSearch(BaseEstimator):

    """Search features contexts using beam search greedy search heuristic"""

    def __init__(self,
                 evaluator: EvaluationFunction,
                 n_processes: int = 5,
                 chunk_size: int = 20,
                 max_context_size: int = 100000,
                 noise_confidence: float = 0.05,
                 noise_factor: float = 0.1,
                 track_all: bool = False,
                 beam_size: int = 100):
        """
        :param beam_size: number of top contexts to keep in each stage
        """
        super(BeamSearch, self).__init__(evaluator, n_processes, chunk_size, max_context_size,
                                         noise_confidence, noise_factor, track_all)
        self._beam_size = beam_size

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

        tracker = ContributionTracker(n_features=len(feature_names), track_all=self._track_all)

        subset_to_eval_cahce: Dict[Tuple[str, ...], float] = {}  # cache results to avoid duplicate evaluations
        context_candidates: List[List[Set[str]]] = [[set()] for _ in feature_names]
        for context_size in tqdm(range(min(len(feature_names), self._max_context_size + 1))):
            noise = pac_noise(len(x), len(feature_names), context_size, self._noise_confidence, self._noise_factor)

            # contexts to eval
            subsets_to_eval = [c for f_cands in context_candidates for c in f_cands
                               if context_to_key(c) not in subset_to_eval_cahce]
            # contexts with feature to eval (for calculating feature contribution)
            subsets_to_eval.extend([c.union({f}) for f, f_cands in zip(feature_names, context_candidates)
                                    for c in f_cands if context_to_key(c.union({f})) not in subset_to_eval_cahce])
            subset_to_eval_cahce.update(self._multiprocess_eval_subsets(subsets_to_eval, x, y, x_test, y_test))
            for f_idx, (f, f_contexts_cands) in enumerate(zip(feature_names, context_candidates)):

                # eval contributions to candidates
                conts = []
                for context in f_contexts_cands:
                    contribution = subset_to_eval_cahce[context_to_key(context.union({f}))] \
                                   - subset_to_eval_cahce[context_to_key(context)]
                    tracker.update_value(feature_idx=f_idx, contribution=contribution, context=context,
                                         noise_tolerance=noise)
                    conts.append((contribution, context))

                # calculate new candidates according to beam size
                argmax_k_contexts = [context for _, context in
                                     sorted(conts, key=lambda it: it[0], reverse=True)][:self._beam_size]
                f_candidates = [c.union({f_s}) for c in argmax_k_contexts
                                for f_s in feature_names if f_s not in c and f_s != f]
                # remove context duplications
                f_candidates = [set(c_t) for c_t in set(context_to_key(c) for c in f_candidates)]
                context_candidates[f_idx] = f_candidates

        return MciValues.create_from_tracker(tracker, feature_names)
