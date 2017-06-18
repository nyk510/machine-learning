# coding: utf-8
__author__ = "yamaguchi"
"""
6.4 ガウス過程の事前分布からのサンプリング
"""

import numpy as np
import matplotlib.pyplot as plt


class Kernel(object):
    """
    カーネル関数
    """

    def __init__(self, theta1, theta2, theta3, theta4, kernel_type="gaussian"):
        """
        kernel = Θ1*exp(-Θ2/2*||x1-x2||^2) + Θ3 + Θ4*<x1|x2>

        :param str kernel_type: カーネルの種類を表す文字列. gaussian | ornstein を指定.
        """
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4
        self.kernel_type = kernel_type

    def _kernel(self, x1, x2):
        if self.kernel_type == "gaussian":
            return np.exp(-self.theta2 / 2. * np.inner(x1 - x2, x1 - x2))
        if self.kernel_type == "ornstein":  # オルンシュタインウーレンベック過程
            return np.exp(-self.theta2 * np.power(np.inner(x1 - x2, x1 - x2), .5))

    def __call__(self, x1, x2):
        """
        calculate kernel
        x1,x2: numpy.array, has same dimension
        """
        val = self.theta1 * self._kernel(x1, x2) + self.theta3 + self.theta4 * (np.inner(x1, x2))
        return val

    def __str__(self):
        s = "type={0.kernel_type}_theta1={0.theta1}_theta2={0.theta2}_theta3={0.theta3}_theta4={0.theta4}".format(self)
        return s


if __name__ == '__main__':

    kernel1 = Kernel(1, 4., 10., 0., kernel_type="ornstein")
    num_of_spans = 200  # 離散化する数 増やせばなめらかさは増しますが、計算コストも増えます。

    gram_matrix = np.identity(num_of_spans)
    x_range = np.linspace(-1, 1, num_of_spans)
    for i in range(num_of_spans):
        for k in range(num_of_spans):
            x1 = x_range[i]
            x2 = x_range[k]
            gram_matrix[i][k] = kernel1(np.array([x1]), np.array([x2]))
    color = ["C{i}".format(**locals()) for i in range(4)]
    for i in range(10):
        y = np.random.multivariate_normal(np.zeros(num_of_spans), gram_matrix, 1)
        plt.plot(x_range, y[0], color[i % len(color)])
    plt.title(str(kernel1))
    plt.show()
