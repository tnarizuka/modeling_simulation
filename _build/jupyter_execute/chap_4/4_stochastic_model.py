#!/usr/bin/env python
# coding: utf-8

# # 確率モデル

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy as sp

# 日本語フォントの設定（Mac:'Hiragino Sans', Windows:'MS Gothic'）
plt.rcParams['font.family'] = 'Hiragino Sans'


# ## 確率モデルとは？
# 
# 確率モデルとは，数理モデルの変数が確率変数となっているモデルであり，自然現象から社会現象まで様々な現象のモデル化に用いられる．
# 例えば，コイン投げは，表と裏が一定の確率で出るという数理モデルで記述することができ，最も基本的な確率モデルである．
# また，株価の変動は，毎時刻ごとに株価の上がり幅と下がり幅が確率的に決まるようなモデル（ランダムウォーク）によって記述することができる．

# ## 乱数の生成
# 
# 本節の内容は，文献{cite}`Odaka2018` を参考にしている．
# 
# ある数列が与えられたとき，数の並び方に規則性や周期性がない場合，この数列の各要素を**乱数**と呼ぶ．
# 乱数を生成するにはいくつかの方法がある．
# 例えば，サイコロを振って出た目は乱数の一種である．
# また，カオスや量子力学的な性質を利用した乱数は**物理乱数**と呼ばれる．
# 
# 一方，コンピュータ上で乱数を使ったシミュレーションを行うためには，アルゴリズムにしたがって乱数を生成する．
# この乱数は，完全にランダムな値ではなく，アルゴリズムが分かれば値を予測することができてしまうため，**疑似乱数**と呼ばれる．
# しかし，疑似乱数でもアルゴリズムを工夫すれば，十分にランダムに見える値を生成することができるため，乱数として扱うことができる．

# ### 一様乱数

# 乱数の中でも，ある範囲の値が同じ確率で出現するような乱数を**一様乱数**と呼ぶ．
# 一様乱数は，乱数の中でも最も基本的な乱数であり，様々な確率モデルの構築に用いられる．

# #### 線形合同法
# 
# 一様乱数を生成するアルゴリズムの中で，最も基本的なものが**線形合同法**である．
# 線形合同法は，いくつかの問題があるため，現在ではあまり使われていないが，乱数の生成アルゴリズムの基礎を理解するために，ここで紹介する．

# 線形合同法のアルゴリズムは以下の通りである．
# 
# ```{admonition} 線形合同法のアルゴリズム
# 1. 以下の条件を満たす整数 $ a, b, M $ を決める：
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
#     ※ $ \% $ は剰余を表す．
# 
# ```
# 
# 以上のアルゴリズムを用いると，$ 0\sim M-1 $ の範囲の乱数を得ることができる．
# また，$ a, b, M $ がいくつかの条件を満たすと，最大周期が $ M $ となることが知られている．
# よって，良い乱数を生成するためには，条件を満たした上で $ M $ が大きい方が良い．
# 例えば，$ M=2^{32} $ の場合には，$ a=1664525,\ b=1013904223 $ が良い値とされている．

# これをPythonで実装すると以下のようになる．

# In[7]:


a, b = 1664525, 1013904223
M = 2**32
seed = 7

x = np.array([seed])
for i in range(50):
    x = np.append(x, (a*x[i] + b) % M)


# In[10]:


# [0, 1)の一様乱数
x_2 = x/M
x_2


# In[9]:


# [-1, 1)の一様乱数
x_3 = 2*x - 1
x_3


# #### より精度の高い一様乱数 
# 
# メルセンヌ・ツイスター法

# ### 任意の確率分布に従う乱数

# #### 逆関数法

# 累積分布関数が $ F(x) $ であるような確率分布に従う乱数を生成するには，一様乱数 $ U $ に対して，$ F^{-1}(U) $ を計算すれば良い．
# これを**逆関数法**と呼ぶ．
# 逆関数法は，累積分布関数の逆関数が求まるような確率分布であれば，どのような確率分布にも適用することができる．

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
# ここで，求めたい確率分布の累積分布関数を $ F(x) $ とし，その逆関数が存在するとする．
# このとき，上の式は
# 
# $$
#     P(F^{-1}(U) \leq F^{-1}(u)) = u
# $$
# 
# と変形できる．
# また，$ F^{-1}(u) = x $ と変数変換すると，$ u = F(x) $ である．
# よって，
# 
# $$
#     P(F^{-1}(U) \leq x) = F(x)
# $$
# 
# 以上より，一様乱数 $ U $ に対して，新たな確率変数を $ X=F^{-1}(U) $ と定義すると，$ X $ は求めたい確率分布 $ F(x) $ に従う．
# ```

# #### 複雑な確率分布の場合
# 
# 正規分布の場合は，中心極限定理（後述）を用いた方法やボックス・ミュラー法が有名．
# また，マルコフ連鎖モンテカルロ法（MCMC）は，マルコフ連鎖に基づいて任意の確率分布に従う乱数を生成する方法であり，様々な分野で利用されている．

# ## 大数の法則と中心極限定理

# ### ベルヌーイ試行

# ### 二項分布

# ### 大数の法則

# ### 中心極限定理

# ## ランダムウォーク
