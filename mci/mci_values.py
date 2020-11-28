import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional
from itertools import combinations
import seaborn as sns
import pandas as pd
from scipy.special import softmax

from mci.estimators.contribution_tracker import ContributionTracker
from mci.utils.estimators_util import context_to_key


class MciValues:

    """contain MCI values and project relevant plots from them"""

    def __init__(self,
                 values: Sequence[float],
                 feature_names: Sequence[str],
                 contexts: Sequence[Tuple[str, ...]],
                 additional_values: Optional[Sequence[Sequence[float]]] = None,
                 additional_contexts: Optional[Sequence[Sequence[Tuple[str, ...]]]] = None,
                 shapley_values: Optional[Sequence[float]] = None):
        """
        :param values: array of MCI values for each feature
        :param feature_names: array of features names (corresponds to the values)
        :param contexts: array of argmax contribution contexts for each feature (corresponds to the values)
        :param additional_values: placeholder for additional MCI values per feature (for non optimal values)
        :param additional_contexts: placeholder for additional MCI contexts per feature (for non optimal values)
        :param shapley_values: shapley values for comparison (optional)
        """
        self.values = values
        self.feature_names = feature_names
        self.contexts = contexts
        self.additional_values = additional_values
        self.additional_contexts = additional_contexts
        self.shapley_values = shapley_values

    @classmethod
    def create_from_tracker(cls, tracker: ContributionTracker, feature_names: Sequence[str]):
        return cls(values=tracker.max_contributions,
                   feature_names=feature_names,
                   contexts=tracker.argmax_contexts,
                   additional_values=tracker.all_contributions,
                   additional_contexts=tracker.all_contexts,
                   shapley_values=tracker.avg_contributions)

    def plot_values(self, plot_contexts: bool = False, score_name="MCI"):
        """Simple bar plot for MCI values per feature name"""
        score_features = sorted([(score, feature, context) for score, feature, context
                                 in zip(self.values, self.feature_names, self.contexts)],
                                key=lambda x: x[0], reverse=True)

        if plot_contexts:
            features = [f"{f} ({', '.join(context)})" for score, f, context in score_features]
        else:
            features = [f for score, f, context in score_features]
        plt.barh(y=features, width=[score for score, f, context in score_features])
        plt.title(f"{score_name} feature importance")
        plt.xlabel(f"{score_name} value")
        plt.ylabel("Feature name")
        plt.show()

    def plot_contexts_affiliation(self):

        f_c_to_cont = {}
        for f_idx, f in enumerate(self.feature_names):
            f_c_to_cont[f] = {}
            for c, v in zip(self.additional_contexts[f_idx], self.additional_values[f_idx]):
                f_c_to_cont[f][context_to_key(c)] = v

        contexts_affiliation = [[0]*len(self.feature_names) for _ in self.feature_names]
        for f_i_idx, f_i in enumerate(self.feature_names[:-1]):
            for f_j in self.feature_names[f_i_idx+1:]:
                f_j_idx = self.feature_names.index(f_j)
                features_exluded = [f for f in self.feature_names if f not in (f_i, f_j)]
                for comb_size in range(len(self.feature_names) - 1):
                    for comb in combinations(features_exluded, comb_size):
                        c_i_j_cont = f_c_to_cont[f_i][context_to_key(set(comb).union({f_j}))]
                        c_i_cont = f_c_to_cont[f_i][context_to_key(comb)]
                        cur_aff = c_i_j_cont - c_i_cont
                        if cur_aff > contexts_affiliation[f_i_idx][f_j_idx]:
                            contexts_affiliation[f_i_idx][f_j_idx] = cur_aff
                            contexts_affiliation[f_j_idx][f_i_idx] = cur_aff

        plt.rcParams.update({'font.size': 50})
        contexts_affiliation = softmax(contexts_affiliation)
        contexts_affiliation = pd.DataFrame(contexts_affiliation, columns=self.feature_names, index=self.feature_names)
        sns.set_theme()
        ax = sns.heatmap(contexts_affiliation, cmap="YlGnBu")
        plt.show()

    def results_dict(self) -> dict:
        results = {
            "feature_names": self.feature_names,
            "mci_values": self.values,
            "contexts": self.contexts,
            "shapley_values": self.shapley_values
        }
        return results
