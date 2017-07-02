# Machine Learning

機械学習に関する python ファイルをまとめていくリポジトリ.

## 環境

* python 3.5

## requirements

* numpy
* scipy
* scikit-learn
* matplotlib
* seaborn

## Contents

### サンプリング

* [gibbs sampling による多次元ガウス分布のサンプリング](gibbs_sampling.py)

### PRML

* [カーネル密度推定](./prml/density_estimation.py)
* ガウス過程 (Gaussian Process)
    * [事前分布](./prml/gaussian_process_prior.py)
    * [新しいデータに対する予測](./prml/gaussian_process_estimation.py)
* [RVM (Relevant Vector Machine)](./prml/rvm.py)
* [EM Algorithm](./prml/em_algorithm.py)
* [Variance Inference](./prml/variance_inference.py)

### その他

* [black sholes 方程式](./black_scholes.py)
