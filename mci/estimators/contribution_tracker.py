from typing import Set, List
import json


class ContributionTracker:

    def __init__(self, n_features: int, track_all: bool = False):
        """
        :param n_features: number of features to track contributions for
        :param track_all: if true, saves all observed contributions and not only max per feature
        """
        self._n_features = n_features
        self.track_all = track_all

        self.max_contributions = [0.0]*self._n_features
        self.sum_contributions = [0.0]*self._n_features
        self.n_contributions = [0.0]*self._n_features
        self.argmax_contexts = [set() for _ in range(self._n_features)]

        self.all_contributions = [[] for _ in range(self._n_features)]
        self.all_contexts = [[] for _ in range(self._n_features)]

    def update_value(self, feature_idx: int, contribution: float, context: Set[str], noise_tolerance: float = 0.0):
        if contribution > self.max_contributions[feature_idx] + noise_tolerance:
            self.max_contributions[feature_idx] = contribution
            self.argmax_contexts[feature_idx] = context

        self.n_contributions[feature_idx] += 1
        self.sum_contributions[feature_idx] += contribution

        if self.track_all:
            self.all_contributions[feature_idx].append(contribution)
            self.all_contexts[feature_idx].append(context)

    def update_tracker(self, tracker: 'ContributionTracker'):
        if self.track_all and tracker.track_all:
            for feature_idx, (f_conts, f_contexts) in enumerate(zip(tracker.all_contributions, tracker.all_contexts)):
                for cont, context in zip(f_conts, f_contexts):
                    self.update_value(feature_idx, cont, context)
        else:
            for feature_idx, (cont, context) in enumerate(zip(tracker.max_contributions, tracker.argmax_contexts)):
                self.update_value(feature_idx, cont, context)

    def save_to_file(self, feature_names: List[str], file_path: str):
        state = {}

        state["max_contributions"] = self.max_contributions
        state["sum_contributions"] = self.sum_contributions
        state["n_contributions"] = self.n_contributions
        state["argmax_contexts"] = [list(c) for c in self.argmax_contexts]
        state["feature_names"] = feature_names

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    @staticmethod
    def load_from_file(file_path: str, feature_names: List[str]) -> 'ContributionTracker':
        with open(file_path) as f:
            state = json.load(f)

        assert feature_names == state["feature_names"]
        tracker = ContributionTracker(n_features=len(feature_names), track_all=False)
        tracker.max_contributions = state["max_contributions"]
        tracker.sum_contributions = state["sum_contributions"]
        tracker.n_contributions = state["n_contributions"]
        tracker.argmax_contexts = state["argmax_contexts"]
        return tracker

    @property
    def avg_contributions(self):
        return [s/max(n, 1) for s, n in zip(self.sum_contributions, self.n_contributions)]