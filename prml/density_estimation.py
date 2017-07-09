# coding: utf-8
"""
カーネル密度推定
@section 2
"""

__author__ = "nyk510"

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class KernelEstimation():
    def __init__(self, h, dist, kernel):
        self.h = h
        self.dist = dist
        self.kernel = kernel

        self.data = None
        self.norm_ratio = None

    def __str__(self):
        s = "kernel: {0.kernel} h: {0.h:.2f}".format(self)
        return s

    def _kernel(self, d):
        """
        return kernel distance
        """
        if self.kernel == "gaussian":
            norm = np.linalg.norm(d)
            return np.exp(- norm ** 2. / (2. * self.h ** 2.))
        return 0

    def _calculate_normalization_ratio(self):
        """
        calculate normalization ratio
        """
        if self.kernel == "gaussian":
            return 1 / np.power((2 * self.h ** 2. * np.pi), (self.dist / 2.)) / self.data.size
        else:
            return 0

    def fit(self, X):
        self.data = X
        self.norm_ratio = self._calculate_normalization_ratio()
        return

    def predict(self, x):
        """
        predict by kernel estimate
        """
        x = np.asarray(x)
        prob = np.zeros_like(x)
        for i in range(x.size):
            for k in range(self.data.size):
                prob[i] += self._kernel(x[i] - self.data[k])
        return prob * self.norm_ratio


if __name__ == '__main__':
    np.random.seed(19)
    rv = multivariate_normal(1, 1)
    x = np.linspace(-5, 5, 100)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(x, rv.pdf(x), "-", color="C0", label="True Density Function")
    samples = rv.rvs(100)  # sampleの生成
    ax1.hist(samples, normed=True, facecolor='C0', alpha=0.3)

    for h in np.logspace(-1., .5, num=5):
        model = KernelEstimation(h, 1., "gaussian")
        model.fit(samples)
        preds = model.predict(x)
        ax1.plot(x, preds, "--", label=str(model))
        ax1.legend()
    fig.savefig("density_estimation.png", dpi=150)
