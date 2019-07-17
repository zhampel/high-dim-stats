from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt


except ImportError as e:
    print(e)
    raise ImportError

# Sample a random standard normal vector
def sample_z(ndims=10, bsize=10):

    zn = np.random.normal(0, 1, (bsize, ndims))

    return zn

# Theoretical M-P distribution of eigenvalues
def f_marcpastur(alpha=0.0):

    # Get eigenvalue max, min limits
    t_a_min = (1 - np.sqrt(alpha))**2
    t_a_max = (1 + np.sqrt(alpha))**2

    # Eigenvalue abscissa, only on min/max support
    gamma = np.linspace(t_a_min, t_a_max, 1000)

    # Unscaled values (corrected from Wainwright Fig. 1.2 caption)
    f_mp = (t_a_max - gamma) * (gamma - t_a_min)
    f_mp = np.sqrt(f_mp)/gamma

    return f_mp, gamma

def main():
    global args
    parser = argparse.ArgumentParser(description="Covariance estimation demo script")
    parser.add_argument("-d", "--num_dims", dest="num_dims", default=10, type=int, help="Data dimensionality")
    parser.add_argument("-n", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    args = parser.parse_args()

    ndims = args.num_dims
    bsize = args.batch_size
    alpha = np.float(ndims)/np.float(bsize)

    # Generate some simulated data
    data = sample_z(ndims=ndims, bsize=bsize)

    # Sample covariance is outer product of data samples
    samp_cov = np.matmul(data.T, data)
    samp_cov /= np.float(bsize)

    # Compute eigenvalues of sample covariance matrix
    evals = np.linalg.eigvals(samp_cov)
    sort_evals = np.sort(evals)

    # Histogram the eigenvalues
    max_eigval_theo = (1 + np.sqrt(alpha))**2
    lims = [0, 1.25*max_eigval_theo]
    edges = np.linspace(lims[0], lims[1], 101)
    cents = np.diff(edges)+edges[:-1] #edges[1:] - edges[:-1]
    hist, edges = np.histogram(evals, bins=edges)

    # Scale by max value
    #sum_hist = np.sum(hist)
    #hist = hist / np.float(sum_hist)
    max_hist = np.max(hist)
    hist = hist / np.float(max_hist)

    # Marcenko-Pastur law
    fmp, gamma_fine = f_marcpastur(alpha=alpha)
    fmp /= np.max(fmp)

    figname = 'eighist_d%i_n%i.png'%(ndims, bsize)
    fig, ax = plt.subplots(figsize=(16,10))
    ax.step(cents, hist, label='Empirical')
    ax.plot(gamma_fine, fmp, '--r', label='Theory')
    ax.set_xlabel(r'Eigenvalue, $\lambda(\hat{\Sigma})$', fontsize=18)
    ax.set_ylabel('Scaled Density', fontsize=18)
    ax.set_title(r'Eigenvalues $(\alpha = %.02f = d/n = %i/%i)$'%(alpha,ndims,bsize), fontsize=24)
    ax.grid()
    #plt.yscale('log')
    plt.legend(loc='upper right', numpoints=1, fontsize=16)
    plt.tight_layout()
    fig.savefig(figname)
    print(figname)


if __name__ == "__main__":
    main()
