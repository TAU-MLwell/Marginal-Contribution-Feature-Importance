from typing import Callable, Optional
import numpy as np
import lightgbm as lgb
import xgboost
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import check_scoring, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.base import is_classifier

from mci.utils.estimators_util import is_empty
from mci.utils.type_hints import MultiVariateArray, UniVariateArray

EvaluationFunction = Callable[[MultiVariateArray, UniVariateArray,
                               Optional[MultiVariateArray], Optional[UniVariateArray]], float]


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


class LgbEvaluator:

    DEAFULT_PARAMS = {
        'objective': 'multiclass',
        'metric': 'auc',
        'is_unbalance': 'true',
        'boosting': 'gbdt',
        'num_leaves': 20,
        'min_data_in_leaf': 4,
        'feature_fraction': 0.2,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'learning_rate': 0.05,
        'verbose': -1,
        'num_class': 4
    }

    def __init__(self, param_dict: dict = DEAFULT_PARAMS, num_boost_round: int = 603, early_stopping_rounds: int = 5):
        self._param_dict = param_dict
        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds

    def train_lgb(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=5)
        lgb_train = lgb.Dataset(x_train, y_train, feature_name=list(x.columns))
        lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)
        gbm = lgb.train(self._param_dict,
                        lgb_train,
                        num_boost_round=self._num_boost_round,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=self._early_stopping_rounds,
                        verbose_eval=False)
        return gbm

    def __call__(self,
                 x,
                 y,
                 x_test,
                 y_test) -> float:
        gbm = self.train_lgb(x, y)
        y_test = y_test.astype(int)
        y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
        auc = roc_auc_score(y_test, y_pred)
        return auc

class xgboostEvaluator:

    def __call__(self,
                 x,
                 y,
                 x_test,
                 y_test) -> float:
        params = {
            "learning_rate": 0.01,
            "n_estimators": 600,
            "max_depth": 4,
            "subsample": 0.5,
            "reg_lambda": 5.5,
            "reg_alpha": 0,
            "colsample_bytree": 1
        }

        xgb_model = xgboost.XGBRegressor(
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],  # math.pow(10, params["learning_rate"]),
            subsample=params["subsample"],
            reg_lambda=params["reg_lambda"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            n_jobs=16,
            random_state=1,
            objective="survival:cox",
            base_score=100
        )

        if x.shape[1] == 0:
            return 0.5

        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=5)

        xgb_model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            # eval_metric="logloss",
            early_stopping_rounds=10,
            verbose=False
        )
        preds = xgb_model.predict(x_test)

        return max(xgboostEvaluator.c_statistic_harrell(preds, y_test.to_numpy()), 0.5)

    @staticmethod
    def c_statistic_harrell(pred, labels):
        total = 0
        matches = 0
        for i in range(len(labels)):
            mask = ((labels > 0) & (labels < abs(labels[i]))).flatten()
            total += mask.sum()
            matches += (pred[mask] > pred[i]).sum()
        return matches / total
