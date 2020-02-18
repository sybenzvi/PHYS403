import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def nSquaredSum(X, Y):
    """Returns negative squared sum of X and Y"""
    return -(X**2 + Y**2)


# Define grid of points over which to operate.
# NOTE: x and y are 2d arrays, this is required for plotting
x, y = np.meshgrid(np.linspace(-10, 10, 101), np.linspace(-10, 10, 101))


# Evaluate function quadratic() at every point in the grid
z = np.zeros_like(x)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        z[i][j] = nSquaredSum(x[i][j], y[i][j])


fig, ax = plt.subplots(1,1, figsize=(8,5))
ax.plot(0, 0, 'x', color='k')
cs = ax.contour(x, y, z, levels=np.linspace(-90, -15, 4), cmap='coolwarm')
ax.clabel(cs, inline=True, fontsize=12)
ax.set(xlabel=r'$X$', ylabel=r'$Y$')
fig.tight_layout()
fig.savefig('contours_example.pdf')
plt.show()


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot 3d surface
# NOTE: THIS IS NOT REQUIRED FOR HW. It is intended only to further illustrate contours.
ax.plot_surface(x, y, z, cmap='coolwarm', alpha=0.5)
for level, contour in zip(np.linspace(-90, -15, 4), cs.allsegs):
    dat0 = contour[0]
    ax.plot(dat0[:, 0], dat0[:, 1], level, color='k')
    ax.text(dat0[0, 0], dat0[0, 1], level, '{0}'.format(level), zdir=(1, 0, 0), ha='center', va='top')

ax.view_init(elev=50, azim=-45)
ax.set(xlabel=r'$X$', ylabel=r'$Y$', zlabel=r'$Z$', title=r'$Z = -(X^2 + Y^2)$')
fig.tight_layout()
fig.savefig('contours_example_3d.pdf')
plt.show()