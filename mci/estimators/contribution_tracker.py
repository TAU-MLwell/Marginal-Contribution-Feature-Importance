from typing import Set


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

    @property
    def avg_contributions(self):
        return [s/max(n, 1) for s, n in zip(self.sum_contributions, self.n_contributions)]
