#!/usr/bin/env python
# coding: utf-8

# # 確率モデル

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy as sp
from scipy.stats import uniform, bernoulli, binom, norm, poisson, expon

# 日本語フォントの設定（Mac:'Hiragino Sans', Windows:'MS Gothic'）
plt.rcParams['font.family'] = 'Hiragino Sans'


# 本章の内容は，文献{cite}`Odaka2018, TokyoUniv2019` を参考にしている．

# <!-- ## 確率モデルとは？ -->
# 
# 確率モデルとは，数理モデルの変数が確率変数となっているモデルであり，自然現象から社会現象まで様々な現象の記述に用いられる．
# 例えば，コイン投げは，表と裏が一定の確率で出るという数理モデルで記述でき，これは最も基本的な確率モデルである．
# また，株価の変動は，毎時刻ごとに株価の上がり幅と下がり幅が確率的に決まるようなモデル（ランダムウォーク）によって記述することができる．
# 本章では，確率モデルをシミュレーションする際に必要な乱数について述べた後，確率モデルにおいて重要な大数の法則と中心極限定理，そしてランダムウォークについて説明する．
# 

# ## 乱数の生成
# 
# ある数列が与えられたとき，数の並び方に規則性や周期性がない場合，この数列の各要素を**乱数**と呼ぶ．
# 乱数を生成するにはいくつかの方法がある．
# 例えば，サイコロを振って出た目を記録するだけでも，1〜6の数値から成る乱数を生成することができる．
# また，カオスや量子力学などのランダムな物理現象を利用した乱数は**物理乱数**と呼ばれ，規則性や周期性がない真の乱数として研究されている．
# 
# 一方，コンピュータ上でアルゴリズムにしたがって乱数を生成する方法もある．
# このように生成された乱数は**疑似乱数**と呼ばれる．
# 疑似乱数はアルゴリズムが分かれば値を予測することができてしまうが，シミュレーションのためには逆にこの性質が長所となる．
# また，アルゴリズム次第で十分にランダムに見える値を高速に生成することができることも長所である．
# 一方，疑似乱数にはある種の周期性が現れてしまうという欠点があるため，これを解決するための様々なアルゴリズムが提案されている．

# ### 一様乱数

# 乱数の中で最も基本的なものは，ある範囲の値が同じ確率で出現する**一様乱数**である．
# 一様乱数は，一様分布に従う確率変数の実現値と捉えることができ，様々な確率モデルの構築に用いられる．

# #### 線形合同法
# 
# 一様乱数を生成するアルゴリズムの中で，最も基本的なものが**線形合同法**である．
# 線形合同法にはいくつかの問題があるため，精度が求められる大規模なシミュレーションには使われないが，手軽に乱数を生成したりアルゴリズムの基礎を理解するためには便利である．

# ```{admonition} 線形合同法のアルゴリズム
# 1. 以下の大小関係を満たすように整数 $ a, b, M $ を決める：
#    
#    $$
#     0 < a < M\\
#     0 \leq b < M
#    $$
#    
# 2. 乱数列の初期値 $ x_0 $ を決める（乱数の**シード**と呼ぶ）．
# 3. 以下の漸化式によって乱数列 $ \{x_n\} $ を生成する：
#     
#     $$
#     x_{n+1} = (ax_{n} + b) \% M
#     $$
#     
#     ここで，$ \% $ は剰余を表す．
# 
# ```
# 
# 以上のアルゴリズムを用いると，$ 0 $ 以上 $ M-1 $ 以下の乱数列を得ることができる．
# また，$ a, b, M $ がいくつかの条件を満たすと，最大周期が $ M $ となることが知られてため{cite}`TokyoUniv2019`，良い乱数を生成するためには条件を満たした上で $ M $ を大きく取る方が良い．
# 例えば，$ M=2^{32},\ a=1664525,\ b=1013904223 $ の場合には，最大周期が $ 2^{32} $ となる．
# 
# なお，$ M $ が偶数の場合には偶数と奇数が交互に出現してしまうなど，実用的には問題も多い．

# **Pythonでの実装**

# まず，アルゴリズムに従って素朴に実装すると以下のようになる．

# In[2]:


a, b, M = 1664525, 1013904223, 2**32

U = np.array([0])
for i in range(10):
    U = np.append(U, (a*U[i] + b) % M)
print(U)


# 次に，$ x_{\mathrm{min}} $ 以上 $ x_{\mathrm{max}} $ 以下の一様乱数を生成する汎用的な関数を作成する．

# In[25]:


# 線形合同法により[0, 1)の一様乱数を生成する
def lcg(seed=1, size=100, umin=0, umax=1):
    a, b, M = 1664525, 1013904223, 2**32

    U = np.array([seed])
    for i in range(size-1):
        U = np.append(U, (a*U[i] + b) % M)

    return umin + (umax - umin) * U / M


# In[26]:


U = lcg(seed=10, size=1000, umin=0, umax=100)


# In[5]:


# 乱数列の相関を調べる
U2 = U.reshape(-1, 2)
fig, ax = plt.subplots(figsize=(3, 3))
ax.scatter(U2[:, 0], U2[:, 1], s=10);


# #### より性能の良い一様乱数生成アルゴリズム
# 
# 線形合同法は，周期性の問題などがあるため，実用的にはより性能の良いアルゴリズムが用いられる．
# 例えば，NumpyやSciPyでは，高速で長い周期を持つ一様乱数を生成するために，[メルセンヌ・ツイスタ](http://www.math.sci.hiroshima-u.ac.jp/m-mat/TEACH/ichimura-sho-koen.pdf)やPCG64などのアルゴリズムを選ぶことができる．
# 
# ※ ライブラリのバージョンアップによって変わる可能性がある．

# In[17]:


# Scipyを用いて[0, 1)の一様乱数を生成する
sp.stats.uniform.rvs(size=10)


# In[24]:


# Numpyを用いて[0, 1)の一様乱数を生成する
rng = np.random.default_rng(123) # シードを123に設定
rng.random(10) 


# ### 任意の確率分布に従う乱数

# #### 逆関数法

# ある確率分布に従う乱数を生成したいとする．
# もし，求めたい確率分布について，その累積分布関数の逆関数が求まるならば，以下の**逆関数法**が適用できる．
# 
# ```{admonition} 逆関数法のアルゴリズム
# 1. 求めたい確率分布の累積分布関数 $ F(x) $ を求める．
# 2. $ F(x) $ の逆関数 $ F^{-1} $ を計算する．
# 3. $ [0, 1) $ の一様乱数 $ U $ に対し， $ F^{-1}(U) $ を計算する．
# ```

# ```{admonition} 逆関数法の証明
# :class: dropdown
# 
# $ [0, 1] $ の範囲の一様乱数を $ U $ とすると，累積分布関数は
# 
# $$
#     P(U \leq u) = u
# $$
# 
# と表される．
# ここで，求めたい確率分布の累積分布関数を $ F(x) $ とし，その逆関数 $ F^{-1} $ が求まったとする．
# このとき，上の式は
# 
# $$
#     P(F^{-1}(U) \leq F^{-1}(u)) = u
# $$
# 
# と変形できる．
# よって，$ F^{-1}(u) = x $ と変数変換すれば，$ u = F(x) $ であるので，
# 
# $$
#     P(F^{-1}(U) \leq x) = F(x)
# $$
# 
# を得る．
# 以上より，一様乱数 $ U $ に対して，新たな確率変数を $ X=F^{-1}(U) $ と定義すると，$ X $ は求めたい確率分布 $ F(x) $ に従う．
# ```

# **例）指数分布**
# 
# 指数分布の確率密度関数は以下のように与えられる：
# 
# $$
#     f(x) = \frac{1}{\lambda} e^{-\frac{x}{\lambda}}
# $$
# 
# よって，その累積分布関数は
# 
# $$
#     F(x) =  1 - e^{-\frac{x}{\lambda}}
# $$
# 
# となり，その逆関数は $ F(x)=u $ と置くと，
# 
# $$
#     F^{-1}(u) = -\lambda \log(1-u)  
# $$
# 
# と求まる．
# これより，[0, 1)の一様乱数 $ u $ に対して，$ -\lambda \log(1-u) $ は指数分布に従うことが分かる．

# 以下は，線形合同法を用いて生成した一様乱数から指数乱数を生成する例である．

# In[27]:


# 線形合同法で[0, 1)の一様乱数を生成する
U = lcg(seed=10, size=1000, umin=0, umax=1)

# 逆関数法で指数乱数に変換する
lmd = 1
R_exp = -lmd*np.log(1-U)


# In[28]:


# ヒストグラムの描画
fig, ax = plt.subplots()
ret = ax.hist(R_exp, bins=50, density=1, color='c', edgecolor='w')

# パラメータlmdの指数分布の描画
x = np.linspace(0, 10, 100)
ax.plot(x, expon.pdf(x, scale=lmd), 'r-');


# #### Box-Muller法（正規分布）
# 
# 正規分布の場合は，中心極限定理（後述）を用いた方法やボックス・ミュラー法が知られている．
# ボックスミュラー法は，$ [0, 1) $ の一様乱数 $ U_{1},\ U_{2} $ から標準正規分布に従う正規乱数 $ Z_{1},\ Z_{2} $ を生成することができる．
# 
# ```{admonition} ボックスミュラー法のアルゴリズム
# 1. $ [0, 1) $ の一様乱数 $ U_1, U_2 $ を生成する．
# 2. 以下の式で一様乱数を変換する：
# 
#     \begin{align*}
#     Z_1 &= \sqrt{-2 \log U_1} \cos(2 \pi U_2)\\
#     Z_2 &= \sqrt{-2 \log U_1} \sin(2 \pi U_2)
#     \end{align*}
# 
# ```

# In[10]:


# 線形合同法で[0, 1)の一様乱数を生成する
U1 = lcg(seed=10, size=10000, umin=0, umax=1)
U2 = lcg(seed=20, size=10000, umin=0, umax=1)

# Box-Muller法で正規乱数に変換する
Z1 = np.sqrt(-2*np.log(U1))*np.cos(2*np.pi*U2)
Z2 = np.sqrt(-2*np.log(U1))*np.sin(2*np.pi*U2)


# In[11]:


fig, ax = plt.subplots(figsize=(4, 3))

# 生成した乱数によるヒストグラムの描画
ret1 = ax.hist(Z1, bins=50, density=1, color='c', edgecolor='w')

# 標準正規分布の描画
x = np.linspace(-5, 5, 100)
ax.plot(x, norm.pdf(x), 'r-');


# #### 複雑な確率分布の場合
# 
# 逆関数法は累積分布関数の逆関数が求まる場合にしか適用できなかったが，どんな確率分布にも適用できる方法として，[棄却法](https://en.wikipedia.org/wiki/Rejection_sampling)がある．
# また，マルコフ連鎖に基づいて任意の確率分布に従う乱数を生成するマルコフ連鎖モンテカルロ法（MCMC）は，主にベイズ推定を始めとして様々な分野で用いられている．

# ## ベルヌーイ過程

# 以下のような試行を**ベルヌーイ試行**と呼ぶ：
# 
# - 1回の試行において，起こりうる事象が2種類しかない
# - 各事象が起こる確率は一定である
# - 各試行は独立である
# 
# 
# 通常は，2種類の事象をそれぞれ成功（1），失敗（0）に対応付けた確率変数 $ U $ を考え，成功確率を $ p $，失敗確率を $ 1-p $ とする．
# このとき，確率変数 $ U $ の従う確率分布は
# 
# $$
# 	P(U=u) = p^{u}(1-p)^{1-u} 
# $$
# 
# となり，これを**ベルヌーイ分布**と呼ぶ．
# 例えば，コイン投げは典型的なベルヌーイ試行である．
# 
# ベルヌーイ試行を繰り返すとき，これを**ベルヌーイ過程**と呼ぶ．
# 多くの基本的な確率分布はベルヌーイ過程を基に導くことができる．
# 以下にいくつかの例を示す．
# 
# - **二項分布**：ベルヌーイ過程において，成功回数が従う確率分布．
# - **正規分布**：ベルヌーイ過程において，試行回数が十分大きい場合の成功回数の分布．
# - **幾何分布**：ベルヌーイ過程において，初めて成功するまでの失敗回数が従う確率分布．
# - **負の二項分布**：ベルヌーイ過程において， $ r $ 回目の成功が起こるまでの失敗回数が従う確率分布．
# - **ポアソン分布**：ベルヌーイ過程において，成功確率 $ p $ が小さく，試行回数 $ n $ が大きいときに $ np=一定 $ の条件の下で成功回数が従う確率分布．

# ### 二項分布
# 
# ベルヌーイ試行を $ n $ 回繰り返すとき，成功回数 $ X=\displaystyle\sum_{i=1}^{n} U_{i} $ を新たな確率変数とする．
# このとき，成功が $ x $ 回，失敗が $ n-x $ 回生じたとすると，その確率分布は**二項分布**
# 
# $$
# 	f(x) = \binom{n}{x}p^{x}(1-p)^{n-x}
# $$
# 
# で与えられる．
# この式において，$ p^{x}(1-p)^{n-x} $ は成功が $ x $ 回，失敗が $ n-x $ 回生じる確率を意味する．
# また，$ \binom{n}{x} $ は $ n $ 個から $ x $ を取り出す組み合わせの数 $ _{n}C_{x} $ を表し，$ n $ 回の中で何回目に成功するかの場合の数に対応する．
# なお，$ n=1 $ の場合はベルヌーイ分布に対応する．
# 
# 二項分布は試行回数 $ n $ と成功確率 $ p $ がパラメータであり，これらによって分布の形が決まる．

# In[13]:


# 試行回数nを変化させた場合の二項分布の変化
fig, ax = plt.subplots()
k = np.arange(0, 20, 1)
for n in [5, 10, 30, 50]:
    ax.plot(k, binom.pmf(k, n=n, p=0.2), '-o', mfc='w', ms=5, label='$n=%s, p=0.2$' % n)

ax.set_xlim(0, 20); ax.set_ylim(0, 0.5)
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$f(x)$', fontsize=15)
ax.legend(numpoints=1, fontsize=10, loc='upper right', frameon=True);


# In[14]:


# 成功確率pを変化させた場合の二項分布の変化
fig, ax = plt.subplots()
k = np.arange(0, 20, 1)
for p in [0.1, 0.2, 0.3, 0.5]:
    ax.plot(k, binom.pmf(k, n=10, p=p), '-o', mfc='w', ms=5, label='$n=10, p=%s$' % p)

ax.set_xlim(0, 10); ax.set_ylim(0, 0.5)
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$f(x)$', fontsize=15)
ax.legend(numpoints=1, fontsize=10, loc='upper right', frameon=True);


# #### 演習問題
# 
# - 確率変数 $ X $ が二項分布に従うとき，その期待値と分散が $ E(X) = np $，分散が $ V(X) = np(1-p) $ となることを示せ．

# ### ポアソン分布
# 
# ベルヌーイ試行を独立に $ n $ 回繰り返すとき，成功確率 $ p $ が小さく，かつ試行回数 $ n $ が大きい場合を考える．
# ただし，極限を取る際に平均値が一定値 $ np=\lambda $ になるようにする．
# このような条件で成功回数 $ X $ が従う分布は，二項分布の式に $ np=\lambda $ を代入し，極限 $ p\to 0,\ n\to \infty $ を取ることで
# 
# $$
# 	f(x) = \frac{\lambda^{x}}{x!} \mathrm{e}^{-\lambda}
# $$
# 
# と求まる．
# これを**ポアソン分布**と呼ぶ．
# ポアソン分布は1つのパラメータ $ \lambda $ だけで特徴づけられ，期待値と分散はともに $ \lambda $ となる．
# 
# ポアソン分布は，一定の期間内（例えば１時間や１日）に，稀な現象（$ p\to 0 $）を多数回試行（$ n\to \infty $）した場合にその発生回数が従う分布である．
# ポアソン分布が現れる例は無数にあり，「1日の交通事故件数」，「1分間の放射性元素の崩壊数」，「1ヶ月の有感地震の回数」，「サッカーの試合における90分間の得点数」などは典型例である．

# 以下は $ np=5 $ に保って $ n $ を大きく，$ p $ を小さくしたときの二項分布（実線）と $ \lambda=5 $ のポアソン分布（o）の比較である．
# $ n=80,\ p=1/16 $ になると，二項分布とポアソン分布はほとんど一致していることが分かる．

# In[73]:


fig, ax = plt.subplots(figsize=(5, 3))
x = np.arange(0, 15, 1)
ax.plot(k, poisson.pmf(x, mu=5), 'o', mfc='w', ms=5, label='$\lambda=5$')
for i in np.arange(4):
    p, n = 1/2**(i+1), 10*2**i
    ax.plot(k, binom.pmf(x, n=n, p=p), '-', mfc='w', ms=5, label='$n=%s, p=1/%s$' % (n, 2**(i+1)))

ax.legend(numpoints=1, fontsize=10, loc='upper left', frameon=True, bbox_to_anchor=(1, 1))
ax.set_xlim(0, 15); ax.set_ylim(0, 0.25)
ax.set_xlabel('$x$', fontsize=15)
ax.set_ylabel('$f(x)$', fontsize=15);


# **$ \lambda $が大きいとき**
# 
# ポアソン分布は $ \lambda \to \infty $ において正規分布に近づくことが知られている．
# 以下は $ \lambda $ を変化させた場合の分布の変化である．
# $ \lambda $ が小さいときには左右非対称な分布となるが，$ \lambda $ が大きくなると左右対称な分布に近づくことが分かる．

# In[11]:


fig, ax = plt.subplots()
k = np.arange(0, 25, 1)
for lmd in [1, 4, 8, 12]:
    ax.plot(k, poisson.pmf(k, lmd, 1), '-o', mfc='w', ms=5, label='$\lambda=%s$'% lmd)

ax.set_xlabel('$x$', fontsize=15); ax.set_ylabel('$f(x)$', fontsize=15)
ax.legend(numpoints=1, fontsize=10, loc='upper right', frameon=True);


# #### 演習問題
# 1. ポアソン分布の期待値と分散が共に $ \lambda $ であることを示せ．
# 2. [score_germany.csv](https://drive.google.com/file/d/13mj2GFFNrj8F75K6FW9SPxfTTSGCXlh8/view?usp=sharing)は，ブンデスリーガの2017-2018シーズンにおける一方のチームの１試合の得点数データである．このデータからヒストグラムを作成し，ポアソン分布によってカーブフィッティングせよ（最小二乗法を用いること）．
# 3. [score_nba.csv](https://drive.google.com/file/d/1kVrg4GjjcE_DC-78U0BArBAglJeiwQgr/view?usp=sharing)は，NBAの2015-16シーズンにおける一方のチームの１試合の得点数データである．このデータからヒストグラムを作成し，ポアソン分布によってカーブフィッティングせよ（最小二乗法を用いること）．
# 4. 2.と3.の結果を比較し，ポアソン分布の特徴を考察せよ．

# ## 大数の法則と中心極限定理
# <!-- 
# ### ベルヌーイ試行
# 
# ### 二項分布
# 
# ### 大数の法則
# 
# ### 中心極限定理
# 
# ## ランダムウォーク -->

# ### 大数の法則

# #### ベルヌーイ過程の場合
# 
# 既にベルヌーイ過程における成功回数 $ X=\displaystyle\sum_{i=1}^{n} U_{i} $ が二項分布に従うことを見たが，ここでは $ X $ を $ n $ で割った標本平均（成功割合）
# 
# $$
# 	T = \frac{X}{n} = \frac{1}{n}\displaystyle\sum_{i=1}^{n} U_{i}
# $$
# 
# を新しい確率変数とする．
# このとき， $ T $ の確率分布を $ g(t) $ とすると，$ g(t) $ も二項分布に従い，$ g(t)=nf(nt) $ の関係にある．

# 以下は成功確率 $ p $ を一定値 $ p=0.2 $ に固定して，試行回数 $ n $ を大きくしたときの標本平均 $ T $ の確率分布である．
# この図を見ると，$ n $ の増加に伴って $ t=0.2 $ の周りに分布が集中するとともに，高さが大きくなる様子が分かる．

# In[ ]:


fig, ax = plt.subplots()
k = np.arange(0, 20, 1)

for n in [5, 10, 30, 50]:
    ax.plot(k/n, n*binom.pmf(k, n=n, p=0.2), '-o', mfc='w', ms=5, label='$n=%s, p=0.2$' % n)

ax.set_xlim(0, 1); ax.set_ylim(0, 8)
ax.set_xlabel('標本平均 $t$', fontsize=12)
ax.set_ylabel('$g(t)$', fontsize=15)
ax.legend(numpoints=1, fontsize=10, loc='upper right', frameon=True);


# 以上のような図の変化を数式で確認する．
# まず，成功回数 $ X $ の期待値と分散はそれぞれ $ E(X)=np,\ V(X)=np(1-p) $ であるから，成功割合 $ T=X/n $ の期待値と分散はそれぞれ $ E(T)=p,\ V(T)=p(1-p)/n $ となる．
# これより，成功割合 $ T=X/n $ の期待値は $ n $ に依らず一定 $ p $ で，分散は $ n $ とともに0に近づくことが分かる．
# これが，成功割合 $ T=X/n $ の分布が試行回数 $ n $ の増加とともに $ p $ の近くに集中する理由である．
# 
# 以上のように，ベルヌーイ過程においては，試行回数 $ n\to \infty $ の極限で成功割合 $ T=X/n $ が理論値 $ p $ に一致する．
# このように，確率変数の標本平均が理論値に一致する性質は**大数の法則**と呼ばれている．

# #### 一般の確率分布の場合

# 大数の法則は，一般の確率分布に従う確率変数列について成り立つ一般的な法則であり，以下のように表される．
# 
# ```{admonition} 大数の法則
# 独立同分布に従う $ n $ 個の確率変数 $ U_{1}, U_{2},\ldots, U_{n} $ に対し，それぞれの期待値を $ E[U_{i}]=\mu $ とする．
# このとき，確率変数列の標本平均 $ \displaystyle\frac{1}{n}\sum_{i=1}^{n}U_{i} $ は $ n\to\infty $ で $ \mu $ に一致する．
# ```

# In[ ]:


nsample=100
U = uniform.rvs(loc=0, scale=2, size=nsample)
N = norm.rvs(loc=2, scale=1, size=nsample)
B = binom.rvs(n=10, p=0.3, size=nsample)
P = poisson.rvs(mu=4, size=nsample)
X_u, X_n, X_b, X_p = [], [], [], []
T = np.arange(1, nsample)
for t in T:
    X_u.append(U[:t].mean())
    X_n.append(N[:t].mean())
    X_b.append(B[:t].mean())
    X_p.append(P[:t].mean())

fig, ax = plt.subplots()
ax.plot(T, np.array(X_u), '-')
ax.plot(T, np.array(X_n), '-')
ax.plot(T, np.array(X_b), '-')
ax.plot(T, np.array(X_p), '-')

ax.set_xlim(0, nsample); ax.set_ylim(0, 6)
ax.set_xlabel('試行回数 $n$', fontsize=12)
ax.set_ylabel('標本平均', fontsize=12);


# ### 中心極限定理
# 
# 前節では，成功割合 $ T=X/n $ の分布が $ n $ を大きくしたときに理論値 $ p $ の周りに集中し，大数の法則が成り立つことを見た．
# また，$ n $ を大きくしていくとき，成功回数 $ X $ や成功割合 $ T $ の分布（いずれも二項分布）が左右非対称から左右対称な形へと変化することも見た．
# 実は，$ n $ を十分大きくしたときに出現する左右対称で滑らかな分布は**正規分布**であり，これは以下の中心極限定理による帰結である：
# 
# ```{admonition} 中心極限定理
# 「同じ確率分布に従う $ n $ 個の確率変数の和の分布が $ n $ を大きくしたときに正規分布に近づく」
# ```
# 
# 今の場合，成功回数を表す確率変数 $ X $ は，成功を1，失敗を0とした確率変数 $ U_{i} $ の和 $ \displaystyle X = \sum_{i=1}^{n} U_{i} $ となっているので，「ベルヌーイ分布に従う $ n $ 個の確率変数の和 $ \displaystyle X = \sum_{i=1}^{n} U_{i} $ の分布が $ n $ を大きくしたときに正規分布に近づく」ということになる．

# <!-- 
# 一般に，正規分布は連続型確率変数$ X $の従う確率分布で，確率密度関数は
# 
# $$
# 	f(x) = \frac{1}{\sqrt{2\pi} \sigma} \exp \left[ - \frac{(x-\mu)^{2}}{2\sigma^{2}} \right]
# $$
# 
# で与えられる．
# ここで，$ \mu $と$ \sigma^{2} $は正規分布の期待値と分散に対応する：
# 
# \begin{align*}
# 	E[X] &= \int_{-\infty}^{\infty} xf(x) dx = \mu \\
# 	V[X] &= \int_{-\infty}^{\infty} (x-\mu)^{2}f(x) dx = \sigma^{2}
# \end{align*}
# 
# 以下では，この形の正規分布を$ N(\mu,\sigma^{2}) $と表す．
# 正規分布$ N(\mu, \sigma^{2}) $は平均$ \mu $と分散$ \sigma^{2} $によって形状が図\ref{fig:normal}のように変わる．
# 特に，平均$ \mu $は分布のピークの位置に対応し，分散$ \sigma^{2} $は分布の広がりを決める．
# 正規分布は中心極限定理が背景にある多くの自然現象や社会現象において観られるので，統計学の理論上最も重要な分布である． -->
