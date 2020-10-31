import numpy as np
from scipy.special import comb


def pac_noise(m_samples, n_features, context_size, delta=0.05, factor: float = 1.0):
    return factor*np.sqrt(2/m_samples*np.log(2*comb(n_features, context_size)/delta))


def approx_comb(n, m):
    n*np.log(n) - m*np.log(m) - (n-m)*np.log(n-m)
