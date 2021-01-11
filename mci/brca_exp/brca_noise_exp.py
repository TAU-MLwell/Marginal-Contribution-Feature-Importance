import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from sklearn.linear_model import LogisticRegression
from mci.estimators.permutation_samplling import PermutationSampling
from mci.evaluators import SklearnEvaluator
import pandas as pd


if __name__ == '__main__':

    n_noise_feautres = int(sys.argv[1])
    n_processes = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    exp_name = "brca_{}_02_noise_features".format(n_noise_feautres)

    data = pd.read_csv(r'noise_genes_02.csv')

    y = data["BRCA_Subtype_PAM50"]

    x = data[data.columns[:n_noise_feautres+10]]

    clf = LogisticRegression(random_state=0, max_iter=1000, C=0.1)
    evaluator = SklearnEvaluator(clf, cv=3, scoring='neg_log_loss')

    estimator = PermutationSampling(evaluator=evaluator, out_dir=exp_name,  n_processes=n_processes,
                                    noise_factor=0.0, chunk_size=2000, n_permutations=2**15,
                                    permutations_batch_size=50)

    result = estimator.mci_values(x=x, y=y)
    result.plot_values(r"{}\mci_values.png".format(exp_name))
    result.plot_shapley_values(r"{}\shapley_values.png".format(exp_name))
