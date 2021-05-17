import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


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
