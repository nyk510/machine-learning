# coding: utf-8
"""
relevant vector machine
"""

__author__ = "nyk510"

import numpy as np
import matplotlib.pyplot as plt


class RVM(object):
    def __init__(self, alpha=1., beta=1, repeat_num=20, kernel="gaussian"):
        self.init_alpha = alpha
        self.init_beta = beta
        self.repeat_num = repeat_num
        self.kernel_name = kernel
        if kernel == "gaussian":
            self.kernel = self.gaussian_kernel
        elif kernel == "poly":
            self.kernel = self.poly_kernel
        else:
            raise NameError("Invalid Kernel Name")

    def __str__(self):
        s = "kernel-{0.kernel_name}".format(self)
        return s

    def gaussian_kernel(self, x1, x2):
        """
        :param x1: 
        :param x2: 
        :return: 
        """
        return np.exp(-(x1 - x2) ** 2) * 10.

    def poly_kernel(self, x1, x2):
        d = 5
        return (1 + np.dot(x1, x2)) ** 5

    def _compute_loglikelihood(self, beta, A):
        C = 1. / beta * np.eye(self.N) + self.Phi.dot(np.linalg.inv(A)).dot(self.Phi.T)
        return -1 / 2. * (
            self.N * np.log(2. * np.pi) + np.log(np.linalg.det(C)) + self.t.T.dot(np.linalg.inv(C)).dot(self.t))

    def fit(self, X, t):
        self.X = np.array(X)
        self.N = len(t)
        self.t = np.array(t)
        Phi = []
        Phi.append(np.ones_like(X))
        for x1 in X:
            Phi.append(self.kernel(x1, X))
        self.Phi = np.array(Phi).T
        sigma_trans = []
        m_trans = []
        alpha_trans = []
        beta_trans = []
        log_likelihoods = []

        # 信頼度 alpha と 分散 beta の初期化
        alpha_trans.append(np.ones(len(t) + 1) * self.init_alpha)
        beta_trans.append(self.init_beta)

        for x in range(self.repeat_num):
            A = np.diag(alpha_trans[-1])
            log_likelihood = self._compute_loglikelihood(beta_trans[-1], A)
            print("log-likelihood\t{log_likelihood}".format(**locals()))
            sigma = np.linalg.inv(A + beta_trans[-1] * self.Phi.T.dot(self.Phi))
            m = beta_trans[-1] * sigma.dot(self.Phi.T).dot(t)
            gamma = 1 - alpha_trans[-1] * np.diag(sigma)
            alpha_n = gamma / (m * m)
            beta_n = (self.N - sum(gamma)) / (np.linalg.norm(self.t - self.Phi.dot(m)) ** 2)

            # 上限を超えた信頼度になったベクトルの信頼度を切り下げる
            alpha_n[alpha_n > 10e+10] = 10e+10

            # logとして配列に保存
            alpha_trans.append(alpha_n)
            beta_trans.append(beta_n)
            m_trans.append(m)
            sigma_trans.append(sigma)
            log_likelihoods.append(log_likelihood)

        self.ms = np.array(m_trans)
        self.sigmas = np.array(sigma_trans)
        self.betas = np.array(beta_trans)
        self.alphas = np.array(alpha_trans)
        self.log_likelihoods = np.array(log_likelihoods)

    def predict(self, x):
        phi_x = np.r_[np.array([1]), self.kernel(x, self.X)]

        avg = self.ms[-1].T.dot(phi_x)
        sigma = 1. / self.betas[-1] + phi_x.T.dot(self.sigmas[-1]).dot(phi_x)
        return avg, sigma

    def get_support_vectors(self, min_max_ratio=.2):
        """
        サポートベクトルを取得します
        
        :param float min_max_ratio:  サポートベクトルの信頼度の割合
        """

        alpha = self.alphas[-1][2::]
        min_alpha = min(alpha)
        upper_limit = min_alpha / min_max_ratio
        support_X = self.X[alpha < upper_limit]
        support_t = self.t[alpha < upper_limit]
        return support_X, support_t


if __name__ == '__main__':

    sample_size = 30
    np.random.seed(71)


    def true_func(x):
        return np.sin(5. * x) + x


    X = np.random.uniform(low=-1.1, high=1.1, size=sample_size)
    np.sort(X)
    t = true_func(X) + np.random.normal(loc=0., scale=.2, size=sample_size)
    rvm = RVM(repeat_num=20, kernel="gaussian")
    rvm.fit(X, t)

    X1 = np.linspace(-1.5, 1.5, 100)
    result = []
    for x in X1:
        avg1, sigma1 = rvm.predict(x)
        result.append([avg1, sigma1])
    result = np.array(result)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(X1, result[:, 0], color="C1", label="Predict Mean")
    ax1.fill_between(X1, result[:, 0] + result[:, 1] ** .5, result[:, 0] - result[:, 1] ** .5, alpha=.2,
                     label="Predict 1 sigma", color="C1")
    ax1.plot(X, t, "o", color="C0", markersize=5, label="Training Data")
    ax1.plot(X1, true_func(X1), "--", label="True Function", color="C0")
    sX, st = rvm.get_support_vectors()
    ax1.plot(sX, st, "o", color="C2", markersize=5, label="Support Vectors")
    ax1.legend()
    ax1.set_xlim(min(X1), max(X1))
    ax1.set_title(str(rvm))
    fig.savefig("rvm_predict.png", dpi=150)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    ax1.bar(range(sample_size + 1), np.log(rvm.alphas[-1]), width=.9)
    ax1.set_title("Relevant (log alpha)")
    ax1.set_xlabel("Data Index")
    ax1.set_ylabel("Value (log-scale)")
    fig.tight_layout()

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    ax1.pcolor(np.log(rvm.alphas).T)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Data Index")
    ax1.set_title("Log Alpha Transition")
    fig.tight_layout()
    plt.show()
