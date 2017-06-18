# coding: utf-8
__author__ = "nyk510"
"""
EM algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


class EMAlgorithm():
    def __init__(self, num_classes, features):
        """
        constructor
        :param int num_classes: 分離するクラスの数. 
        :param np.ndarray features:
            特徴量データ. shape = (n_samples, n_dims)
        """
        self.class_num = num_classes
        self.features = features
        n_samples, dim = features.shape

        self.n_dimensions = dim
        self.n_samples = n_samples

        # 混合係数
        self.responsibility = np.empty((self.n_samples, num_classes), dtype=float)
        self.nu_list = []
        self.sigma_list = []
        self.pi_list = []
        for i in range(num_classes):
            # 平均値はrandom_uniform，分散は単位行列
            self.nu_list.append(np.random.uniform(low=-1, high=1.0, size=dim))
            self.sigma_list.append(np.eye(dim))
            self.pi_list.append(1. / dim)
        self.previous_likelihood = self.compute_log_likelihood()
        self.nu_path = []

    def _e_step(self):
        """
        混合係数の更新
        """
        for n, coodinate in enumerate(self.features):
            resp = np.zeros(self.class_num)  # 初期化
            for k in range(self.class_num):
                var = stats.multivariate_normal(mean=self.nu_list[k], cov=self.sigma_list[k])
                dense = var.pdf(coodinate)
                resp[k] = (self.pi_list[k] * dense)
            self.responsibility[n, :] = resp / sum(resp)
        return

    def _m_step(self):
        """
        do m_step
        m_step is constructed 3 sequences.
        1:  update new nu_list
        2:  update new sigma_list using new nu_list
        3:  update new responsibility using old pi_list
        """
        N_k_list = np.zeros(self.class_num)
        for k in range(self.class_num):
            N_k = sum(self.responsibility[:, k])
            N_k_list[k] = N_k
        # print nk_list,"nk"
        # step1 update nu_list
        self.nu_path.append(self.nu_list[:])
        for k in range(self.class_num):
            new_nu = np.zeros(self.n_dimensions)
            for n in range(self.n_samples):
                # print (self.responsibility[n,k]*self.data[n,:]).shape
                new_nu += self.responsibility[n, k] * self.features[n, :]
            self.nu_list[k] = new_nu / N_k_list[k]

        # step2
        for k in range(self.class_num):
            new_sigma_k = np.zeros((self.n_dimensions, self.n_dimensions))
            for n in range(self.n_samples):
                array_x = (self.features[n, :] - self.nu_list[k])[:, np.newaxis]  # ベクトルの転置計算のためにいったん行列に変形
                # print array_x
                # print array_x.dot(array_x.T)
                new_sigma_k += self.responsibility[n, k] * array_x.dot(array_x.T)
            self.sigma_list[k] = new_sigma_k / N_k_list[k]

        # step3
        for k in range(self.class_num):
            self.pi_list = N_k_list / sum(N_k_list)
        return

    def compute_log_likelihood(self):
        """
        calculate log likelihood. using current nu,sigma,pi,and data
        """
        retval = 0.
        for n in range(self.n_samples):
            inside_log = 0.
            for k in range(self.class_num):
                multi_n = stats.multivariate_normal(mean=self.nu_list[k], cov=self.sigma_list[k])
                dense_x = multi_n.pdf(self.features[n])
                inside_log += dense_x * self.pi_list[k]
            retval += np.log(inside_log)
        print("loglikelihood: {0:.3f}".format(retval))
        return retval

    @property
    def is_terminated(self):
        """
        whether the EM algorithm is terminated or not
        return: boolean
        """
        e = 0.001
        current_likelihood = self.compute_log_likelihood()
        dist = current_likelihood - self.previous_likelihood
        kaizen_ratio = np.abs(dist / current_likelihood)
        # print dist, "kaizen"
        if kaizen_ratio < e:
            print("Complete!")
            return True
        self.previous_likelihood = current_likelihood
        return False

    def run(self, repeat=10):
        """
        lunch algorithm
        doing while is_terminated is true or repeat times over setted repeat number
        """
        for i in range(repeat):
            self._e_step()
            self._m_step()
            if self.is_terminated:
                print("repeat ", i)
                return
        return


def generate_samples():
    sample = np.random.multivariate_normal(mean=[0.5, 0], cov=np.array([[13, -9], [-27, 31]]) / 400., size=100)
    sample2 = np.random.multivariate_normal(mean=[0., 0.], cov=np.array([[5, 3], [3, 5]]) / 40., size=200)
    sample3 = np.random.multivariate_normal(mean=[-1, -0.], cov=np.array([[13, -9], [-27, 31]]) / 400., size=100)
    data = np.vstack([sample, sample2, sample3])
    return data


if __name__ == '__main__':
    data = generate_samples()
    em_alg = EMAlgorithm(3, data)
    em_alg.run(repeat=100)
    plt.scatter(data[:, 0], data[:, 1], s=40, c=em_alg.responsibility, alpha=.5, facecolors='none',
                edgecolor=em_alg.responsibility)
    # plt.plot(np.array(em_alg.nu_list)[:, 0], np.array(em_alg.nu_list)[:, 1], "rh", markersize=10)
    path = np.array(em_alg.nu_path)
    for k in range(em_alg.class_num):
        c = [0, 0, 0]
        c[k] = 1
        plt.plot(path[:, k, 0], path[:, k, 1], "*-", color=c, label="class{k} center".format(**locals()))
    plt.legend()
    plt.show()
