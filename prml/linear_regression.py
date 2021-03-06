# coding: utf-8
"""
Section3
Linear Regression
"""

__author__ = "nyk510"

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class LinearRegression(object):
    """
    シンプルな線形回帰問題をときます。
    Evidence近似による重みパラメータwの分散αと、ノイズパラメータβの最大化も行います。  
    """

    def __init__(self, alpha=.001, beta=1., repeat_num=10):
        """
        :param alpha_list: alpha格納用リスト。alphaはN(w|o,1/alpha*I)で表される精度パラメータ
        :param beta_list: beta格納用リスト。betaはモデルN(t|w^T*phi(x),1/beta)での精度パラメータ
        :param repeat_num: fitを行う際の繰り返し回数
        """
        self.init_alpha = alpha
        self.init_beta = beta
        self.repeat_num = repeat_num

    def fit(self, Phi, t):
        """
        Xとtからパラメータwを推定
        :param array-like Phi
            training data, shape = (n_samples, n_features)
        :param array-like t:
            labels for Phi, shape = (n_samples, ) 
        :return self
        :rtype: LinearRegression
        """

        t = np.array(t)
        N = len(t)
        dim_w = len(Phi[0])
        # 計画行列の固有値計算を先にしておく
        self.eigenvalues = np.linalg.eig(Phi.T.dot(Phi))[0]
        self.eigenvalues[np.abs(self.eigenvalues) < .0001] = 0.  # 情報量が小さい次元の固有値はすべて0だと思うことにする。
        # alphaとbetaを格納する為のリストを用意
        # note:最初はpythonのリストで計算してあとでnumpy.arrayにしたほうが np.append とかよりも早い
        alphas = []
        betas = []
        w = []
        gammas = []
        alphas.append(self.init_alpha)
        betas.append(self.init_beta)

        for i in range(self.repeat_num):
            alpha = alphas[-1]
            beta = betas[-1]
            A = alpha * np.eye(dim_w) + beta * Phi.T.dot(Phi)
            # print "A: ",A
            m_N = np.linalg.solve(A, beta * (Phi.T.dot(t)))
            # print "m_N",m_N
            w.append(m_N)
            gam = sum(self.eigenvalues * beta / (self.eigenvalues * beta + alpha))
            gammas.append(gam)
            new_alpha = gam / sum(m_N * m_N)
            new_beta = (N - gam) / ((t - Phi.dot(m_N)).dot(t - Phi.dot(m_N)))
            # print "a: ",new_alpha,"b: ",new_beta,"gamma: ",gam
            alphas.append(new_alpha)
            betas.append(new_beta)

        self.alpha_list = np.array(alphas)
        self.beta_list = np.array(betas)
        self.w = np.array(w)
        self.gam = np.array(gammas)
        return self

    def predict(self, x):
        """
        :param np.ndarray x:
        :rtype np.ndarray
        """
        return x.dot(self.w)


def compute_polynominal_matrix(X, dimensions):
    """
    単純な多項式による特徴量を返します
    X: numpy.ndarray like
    return: Phi 特徴量ベクトル numpy.ndarray. len(X)*dimensions 行列
    :type dimensions: object
    """
    Phi = [np.ones_like(X)]
    for i in range(dimensions - 1):
        Phi.append(Phi[-1] * X)
    return np.array(Phi).T


def compute_gaussian_phi(X, dim=10, s=1.):
    """
    ガウス固定基底に基づく特徴量matrixを返します.
    固定規定の基準点は [-pi, pi] を dim の数で区切った点.
    :param np.ndarray X:
        変換する特徴量. shape = (n_samples, )

    :param int dim: ガウス基底の次元数
    :param float s: ガウス基底の分散パラメータ
        大きいと分布がゆるやかになります
        要するに、予測分布が遠いデータ点の情報も参照する用になります
    """
    bins = np.linspace(-np.pi, np.pi, num=dim).reshape(1, -1)
    X = X.reshape(-1, 1)
    dist = X - bins
    phi = np.exp(- dist ** 2. / (2 * s ** 2.))
    return np.hstack((phi, np.ones_like(X)))


if __name__ == '__main__':

    # make test data
    sample_num = 20
    true_beta = 10.  # 精度パラメータ
    phi_dim = 20  # 固定基底の数
    np.random.seed(71)
    X = np.random.uniform(-np.pi, np.pi, sample_num)  # -pi~+piまで
    X.sort()
    t = np.sin(X) + np.random.normal(loc=0.0, scale=pow((1 / true_beta), .5), size=sample_num)
    # plt.plot(X,t,"o",label="Training Data")
    # plt.plot(np.linspace(-np.pi,np.pi,200),np.sin(np.linspace(-np.pi,np.pi,200)),"-",label="True Target(No Noize)")
    # plt.show()

    Phi = compute_polynominal_matrix(X, dimensions=phi_dim)
    Phi = compute_gaussian_phi(X, dim=phi_dim, s=1.)

    model = LinearRegression(alpha=1e-5, beta=10., )
    model.fit(Phi, t)

    # predictionグラフの為の特徴量づくり
    X_p = np.linspace(-np.pi, np.pi, 200)
    # Phi_p = compute_polynominal_matrix(X_p,dimentions=phi_dim) #多項式基底関数
    Phi_p = compute_gaussian_phi(X_p, dim=phi_dim, s=1.)  # 今はガウス基底を用います
    print(Phi_p.shape)
    plt.plot(X_p, Phi_p)
    plt.show()

    fig1, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(10, 5))
    ax1 = axes[0]
    ax1.set_title("Maximum Evidence")
    ax1.plot(X_p, model.w[-1].dot(Phi_p.T), "-", label="Prediction", color="C1")
    ax2 = axes[1]
    ax2.set_title("Normal MAP")
    ax2.plot(X_p, model.w[0].dot(Phi_p.T), "-", label="Prediction", color="C1")
    for i in range(2):
        axes[i].plot(np.linspace(-np.pi, np.pi, 200), np.sin(np.linspace(-np.pi, np.pi, 200)), "--",
                     color="C0", label="Truth")
        axes[i].plot(X, t, "o", label="Training data", color="C0", alpha=.8)
        axes[i].set_xlim(-np.pi - .2, np.pi + .2)
        axes[i].set_ylim(-1.5, 1.5)
        axes[i].legend(loc="lower right")
    fig1.tight_layout()
    fig1.savefig("../figures/linear_regression_max-evidence_vs_normal-map.png", dpi=150)
    plt.show()