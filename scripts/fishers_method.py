# Use Fisher's method to compute the p-value of multiple independent p-values.

import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt

def chi2_fisher(p: float, k: int=2) -> (float, float, float):
    """Return the chi-square statistic for combined p-values.

    Parameters
    ----------
    p : float
        An input p-value.
    k : int
        The number of independent p-values.

    Returns
    -------
    chi2: float
        The combined chi-square value, which has 2k degrees of freedom.
    ptot : float
        The equivalent p-value of chi2 with 2k degrees of freedom.
    sigma : float
        The Gaussian sigma (Z-score) of the total p-value.
    """
    x2 = -2*k*np.log(p)
    ptot = chi2.sf(x2, 2*k)
    sigma = norm.isf(ptot)
    return x2, ptot, sigma

# How many independent 3-sigma results does it take to get to 5 sigma?

sigma = 3.
p = norm.sf(sigma)
kmax = 10

fig, ax = plt.subplots(1,1, figsize=(5,5), tight_layout=True)
for k in np.arange(1, kmax+1):
    x2, ptot, sigma = chi2_fisher(p, k)
    print(f'{x2:10g} {ptot:12g} {sigma:8g}')
    ax.scatter(k, sigma, color='tab:blue')

ax.axhline(5, ls=':', color='gray')
ax.set(xticks=np.arange(1, kmax+1),
       xlabel='number of independent tests $k$',
       ylabel='significance [$\sigma$]')
fig.savefig('fishers_method.png', dpi=150)
plt.show()
