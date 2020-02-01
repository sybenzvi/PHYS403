import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def chi2(params, x, y, sigma):
    """Calculate the chi-square for a linear least squares fit.

    Parameters
    ----------
    params : list
        Fit parameters a (slope) and b (intercept).
    x : ndarray
        Data ordinate.
    y : ndarray
        Data coordinate.
    sigma : ndarray or float
        Uncertainties on y. If just one value, assumed equal for all y.

    Returns
    -------
    chi2 : float
        Chi-square sum for a linear fit.
    """
    a, b = params
    return np.sum(((y-(a+b*x)) / sigma)**2)

# Generate fake data with x in [0,1] and y = a + b*x + err, where error is a
# Gaussian of width sigma
a = 1.0
b = 5.0
sigma = 0.3
x = np.linspace(0, 1., 11)
y = a+b*x + sigma*np.random.randn(len(x))

# Initial guess for the fitter:
p0 = [a+0.5, b-0.2]

# Apply the fit. Pass the arrays x, y, and sigma to the function chi2 using the
# parameter args. Use the BFGS method
res = minimize(chi2, p0, args=(x, y, sigma), method="BFGS")
print(res)

# Print best fit parameters and uncertainties
popt = res.x
perr = np.sqrt(np.diag(res.hess_inv))
print("a = %.2f +- %.2f" % (popt[0], perr[0]))
print("b = %.2f +- %.2f" % (popt[1], perr[1]))

# Plot data
plt.errorbar(x, y, yerr=sigma, capsize=0, fmt="k.")

# Plot best fit and a 1 sigma error contour
xfit = np.linspace(0, 1., 101)
yfit = popt[0] + popt[1]*xfit
plt.plot(xfit, yfit, "r-")

ymin = yfit
ymax = yfit
for afit in (popt[0]-perr[0], popt[0]+perr[0]):
    for bfit in (popt[1]-perr[1], popt[1]+perr[1]):
        ymin = np.minimum(ymin, (afit + bfit*xfit))
        ymax = np.maximum(ymax, (afit + bfit*xfit))

plt.fill_between(xfit, ymin, ymax, color="r", alpha=0.2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
