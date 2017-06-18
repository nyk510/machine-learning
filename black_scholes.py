# coding: utf-8
__author__ = "nyk510"
"""
ブラック–ショールズ方程式
https://ja.wikipedia.org/wiki/%E3%83%96%E3%83%A9%E3%83%83%E3%82%AF%E2%80%93%E3%82%B7%E3%83%A7%E3%83%BC%E3%83%AB%E3%82%BA%E6%96%B9%E7%A8%8B%E5%BC%8F
"""

# encoding:UTF-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 定数
t = 0.
S_0 = 100.
K = 100.
r = 0.03
SIGMA = 0.30
T = 1.

# 解析解
d = -(np.math.log(S_0 / K) + ((r - SIGMA * SIGMA / 2.) * T)) / (SIGMA * np.power(T, 0.5))
V_c = S_0 * stats.norm.cdf(-d + SIGMA * np.power(T, .5)) - K * np.exp(-r * T) * stats.norm.cdf(-d)
V_p = K * np.exp(-r * (T - t)) * stats.norm.cdf(d) - S_0 * stats.norm.cdf(d - SIGMA * np.power(T - t, 0.5))


def black_scholes_monte_carlo(repeat):
    rand_n = np.random.normal(size=repeat)
    s_j_t = S_0 * np.exp((r - SIGMA * SIGMA / 2.) * T + SIGMA * rand_n * np.power(T, 0.5))
    return (np.sum(s_j_t[s_j_t > S_0] - S_0)) / float(repeat) * np.exp(-r * T)


def price_monte_carlo(repeat):
    rand_n = np.random.normal(size=repeat)
    s_j_t = S_0 * np.exp((r - SIGMA * SIGMA / 2.) * T + SIGMA * rand_n * np.power(T, 0.5))
    return s_j_t


if __name__ == '__main__':

    print(V_c, "Call Option: Analytical Solution")
    print(V_p, "Put Option: Analytical Solution")
    sol_list = []
    repeat_num = []
    for i in range(15):
        repeat = pow(2, i)
        monte_sol = black_scholes_monte_carlo(repeat)
        print(monte_sol, repeat, "Monte Carlo Solution: d=", V_c - monte_sol)
        sol_list.append(np.abs(V_c - monte_sol) / abs(V_c))
        repeat_num.append(repeat)
    plt.semilogx(repeat_num, sol_list)
    plt.yscale("log")
    plt.xlabel("Num of Repeats")
    plt.ylabel("Error Ratio")
    plt.show()

    sample_num = 10000
    price = price_monte_carlo(sample_num)
    VaR = S_0 - np.percentile(price, 1)
    put_values = price[:]
    put_values[put_values > S_0] = S_0
    put_values = S_0 + S_0 - put_values
    put_VaR = S_0 + V_p - np.percentile(put_values, 1)
    monte_carlo_V_p = np.sum(put_values - S_0) / float(sample_num) * np.exp(-r * T)
    print("Put Option Value(monte carlo):\t", monte_carlo_V_p)
    print("default VaR:\t", VaR, "\r\nput option VaR:\t", put_VaR)
