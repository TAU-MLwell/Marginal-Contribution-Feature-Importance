from typing import Optional
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import check_scoring
from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier

from mci.utils.estimators_util import is_empty
from mci.utils.type_hints import MultiVariateArray, UniVariateArray


class SklearnEvaluator:

    """Creates evaluation function from any SKlearn model"""

    def __init__(self,
                 model,
                 scoring: Optional[str] = None,
                 prior_strategy: Optional[str] = None,
                 cv: int = 3,
                 random_seed: int = 42):
        """
        :param model: an initiated SKlearn model from any type
        :param scoring: a scoring string (see Sklearn docs). if not specified we use model's default
        :param prior_strategy: strategy to evaluate empty set of features (see Sklearn DummyClassifier, DummyRegressor)
        :param cv: number of cross validations to apply per evaluation
        :param random_seed: random seed for DummyClassifier
        """
        self._model = model
        self._prior_model = self._init_prior_model(prior_strategy, random_seed)
        self._cv = cv
        self._scorer = check_scoring(estimator=self._model, scoring=scoring)

    def _init_prior_model(self, prior_strategy: Optional[str], random_seed: int):
        """Init model to evaluate the empty features set performance (i.e predicting using label prior)"""

        if is_classifier(self._model):
            prior_strategy = prior_strategy if prior_strategy else "stratified"
            return DummyClassifier(strategy=prior_strategy, random_state=random_seed)
        else:
            prior_strategy = prior_strategy if prior_strategy else "mean"
            return DummyRegressor(strategy=prior_strategy)

    def __call__(self,
                 x: MultiVariateArray,
                 y: UniVariateArray,
                 x_test: Optional[MultiVariateArray] = None,
                 y_test: Optional[UniVariateArray] = None) -> float:
        """
        :param x: training x data
        :param y: training label data
        :param x_test: test x data (if None will apply cross validation on x, y)
        :param y_test: test label data (if None will apply cross validation on x, y)
        :return: the prediction score of the features in x on y
        """
        model = self._model if len(x) else self._prior_model
        x = x if not is_empty(x) else np.zeros(shape=(len(y), 1))

        if x_test is not None:
            self._model.fit(x, y)
            x_test = x_test if not is_empty(x_test) else np.zeros(shape=(len(y), 1))
            return self._scorer(self._model, x_test, y_test)
        else:
            return cross_val_score(model, x, y, cv=self._cv, scoring=self._scorer).mean()