import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Optional
from mci.estimators.contribution_tracker import ContributionTracker


class MciValues:

    """contain MCI values and project relevant plots from them"""

    def __init__(self,
                 mci_values: Sequence[float],
                 feature_names: Sequence[str],
                 contexts: Sequence[Tuple[str, ...]],
                 additional_values: Optional[Sequence[Sequence[float]]] = None,
                 additional_contexts: Optional[Sequence[Sequence[Tuple[str, ...]]]] = None,
                 shapley_values: Optional[Sequence[float]] = None):
        """
        :param mci_values: array of MCI values for each feature
        :param feature_names: array of features names (corresponds to the values)
        :param contexts: array of argmax contribution contexts for each feature (corresponds to the values)
        :param additional_values: placeholder for additional MCI values per feature (for non optimal values)
        :param additional_contexts: placeholder for additional MCI contexts per feature (for non optimal values)
        :param shapley_values: shapley values for comparison (optional)
        """
        self.mci_values = mci_values
        self.feature_names = feature_names
        self.contexts = contexts
        self.additional_values = additional_values
        self.additional_contexts = additional_contexts
        self.shapley_values = shapley_values

    @classmethod
    def create_from_tracker(cls, tracker: ContributionTracker, feature_names: Sequence[str]):
        return cls(mci_values=tracker.max_contributions,
                   feature_names=feature_names,
                   contexts=tracker.argmax_contexts,
                   additional_values=tracker.all_contributions,
                   additional_contexts=tracker.all_contexts,
                   shapley_values=tracker.avg_contributions)

    def plot_values(self, plot_contexts: bool = False, score_name="MCI", file_path: Optional[str] = None):
        """Simple bar plot for MCI values per feature name"""
        score_features = sorted([(score, feature, context) for score, feature, context
                                 in zip(self.mci_values, self.feature_names, self.contexts)],
                                key=lambda x: x[0])

        if plot_contexts:
            features = [f"{f} ({', '.join(context)})" for score, f, context in score_features]
        else:
            features = [f for score, f, context in score_features]
        plt.barh(y=features, width=[score for score, f, context in score_features])
        plt.title(f"{score_name} feature importance")
        plt.xlabel(f"{score_name} value")
        plt.ylabel("Feature name")

        if file_path:
            plt.savefig(file_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_shapley_values(self, file_path: Optional[str] = None):
        score_features = sorted([(score, feature) for score, feature
                                 in zip(self.shapley_values, self.feature_names)],
                                key=lambda x: x[0])
        features = [f for score, f in score_features]
        plt.barh(y=features, width=[score for score, f in score_features])
        plt.title(f"Shapley feature importance")
        plt.xlabel(f"Shapley value")
        plt.ylabel("Feature name")
        if file_path:
            plt.savefig(file_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def results_dict(self) -> dict:
        results = {
            "feature_names": self.feature_names,
            "mci_values": self.mci_values,
            "contexts": self.contexts,
            "shapley_values": self.shapley_values
        }
        return results
