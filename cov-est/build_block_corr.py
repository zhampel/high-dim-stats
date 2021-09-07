import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm as cm


# Gaussian function
def gaussian(x, mu, sig):
    """
    Gaussian function
    """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))




def plot_corr(corr=None, figname='', title='', comp_hist=None):

    fig = plt.figure(figsize=(18,6))

    ax = fig.add_subplot(121)
    cmap = cm.get_cmap('coolwarm', 30)
    cp = ax.pcolormesh(corr, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_title(r'%s'%title)
    ax.set_xlabel(r'Component $i$')
    ax.set_ylabel(r'Component $j$')
    cbar = fig.colorbar(cp, ax=ax)
    cbar.set_label(r'$\rho_{ij}$', fontsize=16)

    # Plot histogram of off-diag corr
    off_diag = np.ravel(corr[~np.eye(corr.shape[0], dtype=bool)])
    xedges = np.linspace(-1.0, 1.0, 100)
    xcents = (xedges[1:]-xedges[:-1])/2 + xedges[0:-1]
    off_hist = np.histogram(off_diag, bins=xedges)[0]
    off_hist = off_hist / np.sum(off_hist)
    off_hist = np.append(off_hist, off_hist[-1])

    ax = fig.add_subplot(122)
    if comp_hist is not None:
        ax.step(xedges, off_hist, c='r', where='post', label=r'$\rho^{g}$')
        ax.step(xedges, comp_hist, 'k--', where='post', label=r'$\rho^{t}$')

    else:
        ax.step(xedges, off_hist, c='k', where='post', label=r'$\rho^{t}$')

    # Plot fit to normal distribution
    (mu, sigma) = norm.fit(off_diag)
    xfit = np.linspace(-1.0, 1.0, 1000)
    gfit = np.max(off_hist) * gaussian(xfit, mu, sigma)
    ax.plot(xfit, gfit, 'b:', linewidth=2.0, label=r'$\mathscr{N}(%.03f, %.03f)$'%(mu, sigma))

    ax.set_xlabel(r'$\rho(x_i, x_j)$, $i \neq j$')
    ax.set_ylabel(r'Normalized Frequency')
    ax.set_xlim(-1.0, 1.0)
    plt.yscale('log')
    ax.set_ylim(0.001, 1.0)
    #ax.set_ylim(0.0, 0.5)
    ax.grid()
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    #fig.savefig(figname)
    return off_hist, sigma





# Dataset definitions
dim = 50
nblocks = int(dim/2)
mean = np.zeros(dim)

# Covariance matrix defs
sigma_val = 1.0
off_diag = 0.5
block_dim = 2

# Ensure block dimensions are sensical relative to data dimensionality
assert block_dim <= dim and dim % block_dim == 0, \
       "Block dimensions not good: {}. Must be equal to dim {} or multiple of dim.".format(block_dim, dim)

## Block covariance
#cov_block = [[1, 0.5], [0.5, 1]]
##cov = np.kron(np.eye(nblocks), cov_block)
#cov = block_diag(*([cov_block] * nblocks))
        
# Just elements adjacent to diag
if (block_dim == -1):
    cov_block = np.eye(dim, dim, 1) + np.eye(dim, dim, -1)
    cov_block *= off_diag
    cov_block[np.eye(dim, dtype=bool)] = sigma_val
    nblocks = 1
# Block diag structure
else:
    cov_block = off_diag*np.ones((block_dim, block_dim))
    cov_block[np.eye(cov_block.shape[0], dtype=bool)] = sigma_val
    nblocks = int(dim/block_dim)
        
cov = block_diag(*([cov_block] * nblocks))


x = np.random.multivariate_normal(mean, cov, 1000)
print(x.shape)
fig = plt.figure(figsize=(9,6))
mpl.rc("font", family="serif")
ax = fig.add_subplot(111)

ax.plot(x[:,0], x[:,1], 'x')
ax.axis('equal')
plt.show()

test_corr = np.corrcoef(x, rowvar=False)
test_corr_hist, test_fit_sigma = plot_corr(test_corr, 'test_corr.png', title='$X^{t}$ Correlation')
