import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional

from mci.estimators.contribution_tracker import ContributionTracker


class MciValues:

    """contain MCI values and project relevant plots from them"""

    def __init__(self,
                 values: Sequence[float],
                 feature_names: Sequence[str],
                 contexts: Sequence[Tuple[str, ...]],
                 additional_values: Optional[Sequence[Sequence[float]]],
                 additional_contexts: Optional[Sequence[Sequence[Tuple[str, ...]]]]):
        """
        :param values: array of MCI values for each feature
        :param feature_names: array of features names (corresponds to the values)
        :param contexts: array of argmax contribution contexts for each feature (corresponds to the values)
        :param additional_values: placeholder for additional MCI values per feature (for non optimal values)
        :param additional_contexts: placeholder for additional MCI contexts per feature (for non optimal values)
        """
        self.values = values
        self.feature_names = feature_names
        self.contexts = contexts
        self.additional_values = additional_values
        self.additional_contexts = additional_contexts

    @classmethod
    def create_from_tracker(cls, tracker: ContributionTracker, feature_names: Sequence[str]):
        return cls(values=tracker.max_contributions,
                   feature_names=feature_names,
                   contexts=tracker.argmax_contexts,
                   additional_values=tracker.all_contributions,
                   additional_contexts=tracker.all_contexts)

    def plot_values(self, plot_contexts: bool = False):
        """Simple bar plot for MCI values per feature name"""
        score_features = sorted([(score, feature, context) for score, feature, context
                                 in zip(self.values, self.feature_names, self.contexts)],
                                key=lambda x: x[0], reverse=True)

        if plot_contexts:
            features = [f"{f} ({', '.join(context)})" for score, f, context in score_features]
        else:
            features = [f for score, f, context in score_features]
        plt.barh(y=features, width=[score for score, f, context in score_features])
        plt.title("MCI feature importance")
        plt.xlabel("Feature name")
        plt.ylabel("Importance score")
        plt.show()
