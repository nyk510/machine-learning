# coding: utf-8
__author__ = "nyk510"
"""
"""

# coding: UTF-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def gibbs_sampling(nu, cov, sample_size):
    """
    ギブスサンプリングを用いて与えられた共分散, 平均値を持つ
    多次元ガウス分布からのサンプリングを行う関数
    :param np.ndarray nu: 平均値
    :param np.ndarray cov: 共分散
    :param int sample_size: サンプリングする数
    :return:
    :rtype: np.ndarray
    """
    samples = []
    n_dim = nu.shape[0]
    # start point of sampling
    start = [0, 0]
    samples.append(start)
    search_dim = 0

    for i in range(sample_size):
        if search_dim == n_dim - 1:
            """
            search dimension selection is cyclic.
            it can be replaced random choice.
            """
            search_dim = 0
        else:
            search_dim = search_dim + 1

        prev_sample = samples[-1][:]
        A = cov[search_dim][search_dim - 1] / float(cov[search_dim - 1][search_dim - 1])  # A*Σ_yy = Σ_xy
        _y = prev_sample[search_dim - 1]  # previous values of other dimension

        # p(x|y) ~ N(x|nu[x]+A(_y-nu[y]),Σ_zz)
        # Σ_zz = Σ_xx - A0*Σ_yx

        mean = nu[search_dim] + A * (_y - nu[search_dim - 1])
        sigma_zz = cov[search_dim][search_dim] - A * cov[search_dim - 1][search_dim]

        sample_x = np.random.normal(loc=mean, scale=np.power(sigma_zz, .5), size=1)
        prev_sample[search_dim] = sample_x[0]
        samples.append(prev_sample)

    return np.array(samples)


if __name__ == '__main__':
    # 2 dimension normal distribution
    nu = np.ones(2)
    covariance = np.array([[0.5, 0.5], [0.5, 3]])

    # eig_values: 固有値
    # eig_vectors: 固有ベクトル
    eig_values, eig_vectors = np.linalg.eig(covariance)
    average_eigs = np.average(eig_values)
    sample = gibbs_sampling(nu, covariance, 1000)

    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(sample[:, 0], sample[:, 1],
                marker="o", alpha=1., s=30.,
                facecolors='none', edgecolor="C0", label="Samples"
                )

    # 答え合わせ
    # scipy.stats を用いて多次元ガウス分布の確率密度関数を計算
    multi_norm = stats.multivariate_normal(mean=nu, cov=covariance)
    X, Y = np.meshgrid(np.linspace(nu[0] - average_eigs * 2, nu[0] + average_eigs * 2, 100),
                       np.linspace(nu[1] - average_eigs * 2, nu[1] + average_eigs * 2, 100))
    Pos = np.empty(X.shape + (2,))
    Pos[:, :, 0] = X
    Pos[:, :, 1] = Y
    Z = multi_norm.pdf(Pos)
    ax1.contour(X, Y, Z, colors="C0")
    ax1.legend()
    fig.tight_layout()
    fig.savefig("gibbs_sampling.png", dpi=150)
