# Marginal-Contribution-Feature-Importance
source code for Marginal Contribution Importance (MCI) method published in ICML 2021.

To install the package run `pip install Marginal-Contribution-Feature-Importance`

To evaluate MCI an evaluator object need to be initialized. This object defines the calculation 
of the  evaluation function &nu; described in the paper. To initialize an evaluator object using
any scikit-learn model, use the following code (further initialization options can be found in
`SklearnEvaluator` documentation):

```
from sklearn.ensemble import GradientBoostingClassifier
from mci.evaluators.sklearn_evaluator import SklearnEvaluator

model = GradientBoostingClassifier()
evaluator = SklearnEvaluator(model)
```

To calculate the MCI score using the permutation sampling algorithm defined in the paper,
run the following code and provide the number of permutations you would like to sample 
(for multiprocessing with n processes call with `n_processes=n`):

```
from mci.estimators.permutation_samplling import PermutationSampling

mci_estimator = PermutationSampling(evaluator, n_permutations=2**5)
```

Now, to evaluate the MCI score simply call the estimator with X, y pair, where X is a 
dafarame of features and y is a series of label values

`score = mci_estimator.mci_values(X, y)`

To get the MCI scores as an array call `score.mci_values` 
and to plot the feature importance call `score.plot_values()`

