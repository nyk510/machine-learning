# coding: utf-8
__author__ = "nyk510"
"""
10-3 変分ベイズ法による線形回帰モデル
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(71)


def t_func(x):
    """正解ラベルを作る関数
    x: numpy array.
    return t: target array. numpy.array like.
    """
    t = np.sin(x * np.pi)
    # t = np.where(x > 0, 1, -1)
    return t


def plot_target_function(x, ax=None, color="default"):
    """target関数（ノイズなし）をプロットします
    """
    if ax is None:
        ax = plt.subplot(111)
    if color is "default":
        color = "C0"
    ax.plot(x, t_func(x), "--", label="true function", color=color, alpha=.5)
    return ax


def phi_poly(x):
    dims = 3
    return [x ** i for i in range(0, dims + 1)]


def phi_gauss(x):
    bases = np.linspace(-1, 1, 5)
    return [np.exp(- (x - b) ** 2. * 10.) for b in bases]


def qw(alpha, phi, t, beta):
    """
    wの事後分布を計算します。
    変分事後分布はガウス分布なので決定すべきパラメータは平均と分散です
    w ~ N(w| m, S)
    return ガウス分布のパラメータ m, S
    """
    S = beta * phi.T.dot(phi) + alpha * np.eye(phi.shape[1])
    S = np.linalg.inv(S)
    m = beta * S.dot(phi.T).dot(t)
    return m, S


def qbeta(mn, Sn, t, Phi, N, c0, d0):
    """
    betaの変分事後分布を決めるcn,dnを計算します
    変分事後分布はガンマ分布なので決定すべきパラメータは2つです
    beta ~ Gamma(beta | a, b)
    return ガンマ分布のパラメータ a,b
    """
    cn = c0 + .5 * N
    dn = d0 + .5 * (np.linalg.norm(t - Phi.dot(mn)) **
                    2. + np.trace(Phi.T.dot(Phi).dot(Sn)))
    return cn, dn


def qalpha(w2, a0, b0, m):
    """
    alphaの変分事後分布を計算します。
    変分事後分布はガンマ分布ですから決定すべきパラメータは2つです
    alpha ~ Gamma(alpha | a, b)
    return a, b
    """
    a = a0 + m / 2.
    b = b0 + 1 / 2. * w2
    return a, b


def fit(phi_func, x, update_beta=False):
    xx = np.linspace(-2, 2., 100)
    if phi_func == "gauss":
        phi_func = phi_gauss
    elif phi_func == "poly":
        phi_func == phi_poly
    else:
        if type(phi_func) == "function":
            pass
        else:
            raise Exception("invalid phi_func")
    Phi = np.array([phi_func(xi) for xi in x])
    Phi_xx = np.array([phi_func(xi) for xi in xx])

    # 変分事後分布の初期値
    N, m = Phi.shape
    mn = np.zeros(shape=(Phi.shape[1],))
    Sn = np.eye(len(mn))
    beta = 10.
    alpha = .1
    a0, b0 = 1, 1
    c0, d0 = 1, 1

    pred_color = "C1"

    freq = 2
    n_iter = 3 * freq
    n_fig = int(n_iter / freq)

    fig = plt.figure(figsize=(3 * n_fig, 4))
    data_iter = []
    data_iter.append([alpha, beta])

    for i in range(n_iter):
        print("alpha:{alpha:.3g} beta:{beta:.3g}".format(**locals()))

        mn, Sn = qw(alpha, Phi, t, beta)
        w2 = np.linalg.norm(mn) ** 2. + np.trace(Sn)
        a, b = qalpha(w2, a0, b0, m)
        c, d = qbeta(mn, Sn, t, Phi, N, c0, d0)

        alpha = a / b

        if update_beta:
            # betaが更新される
            beta = c / d

        data_iter.append([alpha, beta])

        if i % freq == 0:
            k = int(i / freq) + 1
            ax_i = fig.add_subplot(1, n_fig, k)
            plot_target_function(xx, ax=ax_i)
            ax_i.plot(x, t, "o", label="data", alpha=.8)

            m_line = Phi_xx.dot(mn)
            sigma = (1. / beta + np.diag(Phi_xx.dot(Sn).dot(Phi_xx.T))) ** .5
            ax_i.plot(xx, m_line, "-", label="predict-line", color=pred_color)
            ax_i.fill_between(xx, m_line + sigma, m_line - sigma, label="Predict 1 sigma", alpha=.2, color=pred_color)
            ax_i.set_title(
                "n_iter:{i} alpha:{alpha:.3g} beta:{beta:.3g}".format(**locals()))
            ax_i.set_ylim(-2, 2)
            ax_i.set_xlim(-1.5, 1.5)
        if i == 0:
            ax_i.legend(loc=4)

    fig.tight_layout()
    return fig, data_iter


if __name__ == "__main__":
    n_samples = 20
    x = np.random.uniform(-1, 1, n_samples)
    noise = np.random.normal(scale=1., size=n_samples)
    t = t_func(x) + noise

    plt.figure(figsize=(4, 4))
    xx = np.linspace(-1, 1., 100)
    plot_target_function(xx)
    plt.plot(x, t, "o", label="data")
    plt.ylim(-3, 3)
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig("data.png", dpi=200)

    fig, data_iter = fit(phi_func="gauss", x=x, update_beta=False)
    fig.savefig("iter_not update_beta.png", dpi=200)
    fig, data_iter = fit(phi_func="gauss", x=x, update_beta=True)
    fig.savefig("iter_update_beta.png", dpi=200)

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    pd.DataFrame(data_iter, columns=["alpha", "beta"]).plot(ax=ax1)
    fig.savefig("alpha_beta_trans.png", dpi=200)
