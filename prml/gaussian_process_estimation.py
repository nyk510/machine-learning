# coding: utf-8
"""
ガウス過程による予測を行うスクリプト
"""

__author__ = "nyk510"

import numpy as np


class GaussianKernel(object):
    def __init__(self, theta1=9., theta2=4., theta3=1., theta4=1.):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4

    def __call__(self, x1, x2):
        '''Calculate Kernel.

        @x1: numpy array. shape (m,).
        @x2: numpy array. shape (m,).
            has the same dimension m of x1.
        '''

        return self.theta1 * np.exp(-self.theta2 / 2 * np.linalg.norm(x1 - x2) ** 2) + self.theta3 + self.theta4 * (
            x1 * x2).sum()


class GaussianProcess(object):
    def __init__(self, kernel='gaussian', beta=10.):
        '''
        @kernel: string or GaussianKernel instance.
        @beta: float.
            shrinkage parameter.
            when beta is larger, the predict is more conservative.
        '''
        self.beta = beta

        if kernel == 'gaussian':
            self.kernel = GaussianKernel()
        elif isinstance(kernel, GaussianKernel):
            self.kernel = kernel
        else:
            raise TypeError("Invalid type kernel.")

    def fit(self, X, t):
        """Fit parameter
        calculate data's kernel and it's inverse.

        @X: input data xs.
        @t: input data target values. like numpy.array.
        """
        self.X = X
        self.t = t
        C = []
        for x_i in X:
            c = []
            for x_k in X:
                c.append(self.kernel(x_i, x_k))
            C.append(c)

        C = np.array(C)
        print(C.shape)

        # ここがガウス過程での予測のボトルネック: O(N^3)
        self.C_n_inv = np.linalg.inv(np.array(C) + np.eye(len(X)) * self.beta)

        return

    def predict(self, x):
        """predict value
        :param x: 
            input value for predict
        :return (m(x),sigma(x))
            tuple of predict value's mean and sigma
        :rtype tuple
        """

        k = []

        # fitに使用したデータとのカーネルを計算
        for x_i in self.X:
            k.append(self.kernel(x, x_i))
        k = np.array(k)

        m_x = k.T.dot(self.C_n_inv).dot(self.t)
        variance = self.kernel(x, x) + self.beta - \
                   k.T.dot(self.C_n_inv).dot(k)

        return m_x, variance ** .5


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    sample_size_max = 40
    true_function = np.sin

    np.random.seed(71)
    x = np.random.uniform(-1, .7, sample_size_max)
    train_t = np.sin(x * np.pi) + np.random.normal(scale=.3, size=sample_size_max)
    train_dataset = [x, train_t]
    plt.plot(x, train_t, "o", label='Training Data', color="C0")
    plt.plot(np.linspace(-1, 1, 100), true_function(np.linspace(-1, 1, 100) * np.pi), "--", label='True Function',
             color="C0")
    plt.legend(loc=2)
    plt.title("training data")

    # training data の保存
    # plt.savefig('True_function_and_training_data.png', dpi=150)
    plt.show()

    model = GaussianProcess(beta=.1)
    samples_split = [1, 4, 8, 16, 32]
    plt.figure(figsize=(len(samples_split) * 3, 4))
    for i, sample_num in enumerate(samples_split):
        x = train_dataset[0][:sample_num]
        t = train_dataset[1][:sample_num]
        model.fit(x, t)
        Xs = np.linspace(-1, 1, 100)
        pred_t = []
        for xx in Xs:
            pred_t.append(model.predict(xx))
        pred_t = np.array(pred_t)

        plt.subplot(1, len(samples_split), i + 1)
        plt.plot(x, t, "o", label='Training Data', color="C0")
        plt.plot(np.linspace(-1, 1, 100), true_function(np.linspace(-1, 1, 100) * np.pi), "--", label='True Function',
                 color="C0")
        plt.plot(Xs, pred_t[:, 0], "-", label='Predict', color="C1")
        plt.fill_between(Xs, pred_t[:, 0] + pred_t[:, 1], pred_t[:, 0] - pred_t[:, 1], alpha=.2,
                         label=r'Predict $1-\sigma$ line', color="C1")
        plt.xlim(-1, 1)
        plt.ylim(-2, 2)
        plt.title("Data size: {0}".format(sample_num))
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig('../figures/gp-estimation_by_training_data.png', dpi=150)
    plt.show()
