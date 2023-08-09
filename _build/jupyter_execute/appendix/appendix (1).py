#!/usr/bin/env python
# coding: utf-8

# # 付録

# ## 便利な機能
# 
# ### Jupyterのconfigファイル
# - ターミナルで以下を実行する
#   ```
#   jupyter notebook --generate-config
#   ```
#     - `C:\Users\username\.jupyter`の中に`jupyter_notebook_config.py`というファイルができる．
# - `jupyter_notebook_config.py`を開いて以下を追加
#   ```
#   `c=get_config()`
#   `c.NotebookApp.notebook_dir="起動ディレクトリのパス"`
#   ```
# - これにより，Jupyter Labを起動したときに指定したフォルダが開かれる

# ### Ipythonのプロファイル
# 
# Ipythonプロファイルを作成すると，jupyterの起動時に自動実行したいコマンドを設定できる．
# 
# - ターミナルで以下を実行する
#     ```
#     ipython profile create profile_name
#     ```
#     - `C:\Users\username\.ipython\prifile_name`に`startup`フォルダが作成される．
# - `startup`フォルダの中に`00.ipy`というファイル（スタートアップスクリプト）を作り，自動実行したいコマンドを記述する．
# - 例えば，以下はよく使うので自動importしておくと良い
# 
#     ```python
#     import os
#     import sys
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import pandas as pd
#     ```
# - 自作のモジュール（例えば`my_module.py`）をimportして使う場合，`my_module.py`を一度jupyterでimportした後に，ローカルで`my_module.py`を変更することがよくある．このとき，ローカルで行った変更内容はjupyter側には自動で反映されない．そこで，スタートアップスクリプトに以下を加えておくと自作モジュールの変更が自動で反映される．
#   
#     ```
#     %load_ext autoreload
#     %autoreload 2
#     %matplotlib inline
#     ```

# 例として，`modeling_simulation`フォルダの中に`module`フォルダを作り，以下のプログラムを`my_module.py`として保存する．
# 
# ```python
# def my_func():
#     for i in range(5):
#         print("test%s" % i)
# 
# if __name__ == '__main__':
#     my_func()
# ```
# つまり，このPythonスクリプトのパスは`C:\Users\username\OneDrive\modeling_simulation\module\my_module.py`となる．

# これを単にPythonスクリプトとして実行すると，`if __name__ == '__main__':`以下のコマンドが実行される：

# ```python
# %run "./module/my_module.py"
# ```

# 一方，これをモジュールとしてインポートするには以下のようにする：

# ```python
# import module.my_module as mm
# ```

# この状態で`my_module`内の関数`my_func()`を以下のように`mm.my_func()`として実行できる：

# ```python
# mm.my_func()
# ```

# スタートアップスクリプト内にautoreloadの設定を書いている場合は，ローカルで`my_module.py`を書き換えたら即座に変更内容が反映されるはずである．

# ```python
# mm.my_func()
# ```

# ## Google Colab
# 
# Google Colab（正式名称はGoogle Colaboratoty）はgoogleが提供するPython実行環境であり，Jupyter Notebookがベースになっている．
# 実際，Google Colabで作成したノートブックは".ipynb形式"で保存されるので，相互互換性がある．
# Google Colabの特徴は以下の通りである：
# 
# - ブラウザ上で動作する
# - 基本操作はJupyter Notebookと似ている（細かい操作方法は異なる）
# - 作成したノートブックはGoogle Drive上に保存される
#     - Google Driveが必要（なのでGoogle アカウントも必要）
# - pythonの環境構築が不要（新たにモジュールをインストールすることも可能）
# - 無料でGPUを使用可能
# 
# 特に，Jupyter Notebookの場合は自分のPC上にpython環境を構築する必要があるが，Google Colabはその必要がない点がメリットである．
# また，GPUが無料で使用可能なので，重い計算を行う際にも重宝する．
# 本講義では，基本的にJupyter Labを用いるが，Google Colabを用いても問題ない．

# ### Google colabでjupyter notebookを開く
# 
# - Google Driveを開いて作業フォルダに移動
# - 既存の`.ipynbファイル`を選択するとGoogle Colabが開く
# - 新規作成作成の場合は以下
#     - ［右クリック］→［その他］→［Google Colaboratory］

# ### 必要なモジュールをimportする
# 
# - google colabにインストールされていないモジュール（japanize_matplotlibなど）
# 
#     ```python
#     !pip install japanize-matplotlib
#     import japanize_matplotlib
#     ```
# - 既にインストールされているモジュール
# 
#     ```python
#     import numpy as np
#     ```

# ### Google Driveをマウントする
# 
# Google Driveに保存した自作モジュールやファイルにアクセスしたい場合はGoogle Driveをマウントする必要がある．
# 
# - 以下を実行する
#   
#     ```python
#     from google.colab import drive
#     drive.mount('/content/drive')
#     ```
# - 「このノートブックにGoogleドライブのファイルへのアクセスを許可しますか？」と聞かれるので「Google ドライブに接続」を選択
# - 自分のGoogleアカウントを選択し，「許可」を選択

# ### （任意）自作モジュールをimportする
# 
# ```python
# import sys
# sys.path.append('/content/drive/My Drive/***')
# 
# import ***.xxx
# ```
# ※ なお，自作モジュールの変更を反映したい場合は［ランタイムを出荷時設定にリセット］し，再度マウントする

# ### （任意）matplotlibのスタイルファイルを読み込む
# 
# ```python
# import matplotlib.pyplot as plt
# plt.style.use('/content/drive/My Drive/***/matplotlibrc')
# ```

# ## 感染症の流行
# 
# ### 感染症のモデル化
# 
# これまでの歴史において，人類は天然痘，ペスト，結核，コレラ，エイズ，エボラ出血熱，といった様々な感染症の流行を経験し，現在もなお新型コロナウイルスとの闘いが続いている．
# 感染症の流行は複雑な現象であるが，そのエッセンスは細菌やウイルスが人から人へ伝播するということである．
# そこで，感染症を伝播現象と捉えてそのプロセスを単純化すれば，感染者数の大まかな増減を比較的単純な微分方程式で記述することができる．
# 
# 通常，感染症の数理モデルでは，以下のように複数の状態を考える（これらの他にも，自身が感染しているが他者に移す可能性がない潜伏期の状態（状態 $ E $ ）など，様々な状態を考えることができる．）：
# 
# - 状態$ S $：自身が感染する可能性のある状態（Susceptible or Suspicious）
# <!-- - 状態$ E $：自身が感染しているが，他者に移す可能性がない状態（Exposed） -->
# - 状態$ I $：自身が感染していて，他者に移す可能性がある状態（Infected）
# - 状態$ R $：感染から回復し，他者に移す可能性がない状態（Recovered）
#   
# その上で，時刻 $ t $ においてそれぞれの状態にいる人口の割合を $ S(t),\ I(t),\ R(t) $ とし，これらの変数が従う微分方程式を定める．
# ただし，常にこれら全ての状態を考慮するわけではなく，例えば最も簡単なモデルでは状態 $ S $ と状態 $ I $ だけを取り入れる．
# 以上により，微分方程式を解けば，それぞれの状態にいる人口がどのように時間変化するかが分かるので，感染症の流行を記述することができる．
# 
# なお，通常は微分方程式に加え，全ての変数の和が1（例えば，$ S(t)+I(t)+R(t)=1 $）という関係が成り立つと仮定する．
# この関係式は，総人口を $ M $ としたときに $ MS(t)+MI(t)+MR(t)=M $ と書けるので，人口が常に一定であるという条件を表す．
# これは，考えるシステムにおいて，感染症以外の理由による人口変動が無視できることを案に仮定している．
# 
# 以下では，感染症の数理モデルの中でも単純でよく知られた３つのモデルを紹介する．
# 
# ### SIモデル
# 
# まず，未感染者（状態 $ S $）が感染者（状態 $ I $）と接触すると，一定の確率で感染者（状態 $ I $）に変わるというモデルを考える．
# ただし，一度感染した者はずっと感染したままとする（つまり，最終的に全員が状態 $ I $ になる）．
# このようなモデルは感染症モデルの中で最も単純なモデルであり，**SIモデル**と呼ばれる．
# 
# SIモデルのダイナミクスは，$ S(t) $ が一定確率で $ I(t) $ に変わるというものであり，短い時間間隔 $ \Delta t $ における $ S(t) $ の変化は以下の式で与えられる：
# 
# $$
# 	S(t+\Delta t) = S(t) - \gamma S(t)I(t)\Delta t
# $$
# 
# ここで，右辺第２項は $ S(t) $ の減少分であり，$ S(t)I(t) $ は未感染者と感染者の接触率，$ \gamma $ は感染確率を表す．
# すなわち，未感染者（状態 $ S $）と感染者（状態 $ I $）が接触すると確率 $ \gamma $ で未感染者の割合が減ることを意味する．
# この式の両辺を $ \Delta t $ で割って $ \Delta t\to 0 $ の極限を取ると，
# 
# $$
# 	\frac{dS}{dt} = -\gamma S(t)I(t)
# $$
# 
# が得られる．
# これが未感染者の割合 $ S(t) $ の変化を記述する微分方程式である．
# 
# 一方，感染者の割合 $ I(t) $ については，$ S(t) $ の減少分がそのまま増えるので，
# 
# $$
# 	\frac{dI}{dt} = \gamma S(t)I(t)
# $$
# 
# が成り立つ．
# 
# 以上をまとめると，SIモデルは次のような連立微分方程式として記述される：
# 
# \begin{align}
# 	\left \{
# 	\begin{aligned}
# 		\frac{dS}{dt} &= -\gamma S(t)I(t) \\[10pt]
# 		\frac{dI}{dt} &= \gamma S(t)I(t)
# 	\end{aligned}
# 	\right.
# \end{align}
# 
# ただし，$ S(t)+I(t)=1 $　という関係式が成り立っていることを考慮すると，２つの微分方程式から　$ S(t) $　を消去することができ，$ I(t) $　だけから成る以下の微分方程式に変形できる：
# 
# $$
# 	\frac{dI(t)}{dt} = \gamma (1-I(t))I(t)
# $$
# 
# さらに，この式の両辺に総人口 $ M $ を掛けると
# 
# $$
# 	\frac{dMI(t)}{dt} = \gamma \left(1-\frac{MI(t)}{M}\right)MI(t)
# $$
# 
# となる．
# これは，感染者数 $ MI(t) $ に対する微分方程式であるが，ロジスティックモデルと全く同じ形をしている．
# よって，感染者数は最終的に総人口 $ M $ に収束するが，その増え方は指数関数的な急増から徐々に緩やかなる．
# 
# 
# ### SIRモデル
# 
# SIモデルは一度感染したらずっと感染したままという非現実的なモデルであった．
# しかし，多くの感染症では感染状態（$ I $）になってから時間が経つと，免疫を獲得してそれ以上感染せず，かつ他者に移す可能性もない状態（$ R $）になることが多い．
# そこで，$ S(t),\ I(t) $ に加えて新たな変数 $ R(t) $ も加えた以下のモデルを考える：
# 
# \begin{align}
# 	\left\{
# 	\begin{aligned}
# 		\frac{dS}{dt} &= - \gamma S(t)I(t) \\[10pt]
# 		\frac{dI}{dt} &= \gamma S(t)I(t) - \lambda I(t) \\[10pt]
# 		\frac{dR}{dt} &= \lambda I(t)
# 	\end{aligned}
# 	\right .
# \end{align}
# 
# このモデルは**SIRモデル**と呼ばれ，KermackとMcKendrickによって1927年に提案された．
# SIRモデルは感染症の標準的な数理モデルとしてよく用いられている．
# 
# SIRモデルにおいて，第１式はSIモデルと全く同じである．
# 一方，感染者の割合 $ I(t) $ に対する式（第２式）には新たに第２項 $ -\lambda I(t) $ が加わっている．
# これは，感染者（状態$ I $）が一定の確率 $ \lambda $ で免疫獲得者（状態$ R $）に変化して $ I(t) $ が減少することを表している．
# また，免疫獲得者の割合 $ R(t) $ は $ I(t) $ の減少分だけ増えるので，これを表したのが第３式である．
# なお，これまでと同様に $ S(t)+I(t)+R(t)=1 $ も成り立っている．
# 
# SIRモデルは3変数の連立微分方程式であり，これを一般的に解くのは難しいので通常はコンピュータによる数値計算を行う．
# 一方，感染症が拡大するかどうか（つまり $ I(t) $ が増えるかどうか）は微分方程式を解かずに評価することができる．
# まず，感染者の割合 $ I(t) $ の時間変化を記述する第２式だけを取り出す：
# 
# $$
# 	\frac{dI}{dt} = \gamma S(t)I(t) - \lambda I(t)
# $$
# 
# この式は $ I(t) $ の増加速度 $ dI(t)/dt $ を表す式なので，右辺が正であれば $ I(t) $ は増加，負であれば減少する．
# これは，$ S(t)\gamma /\lambda $ が1より大きければ増加，小さければ減少と言い換えることもでき，このときに現れる量
# 
# $$
# 	R_{t} = S(t)\frac{\gamma}{\lambda}
# $$
# 
# は**実効再生産数**と呼ばれている．
# 実効再生産数 $ R_{t} $ は「感染者（$ I $状態）が回復するまで（$ R $状態になるまで）に平均的に感染させる数」という意味を持つことが示されている．
# 
# $ R_{t} $ を実際の感染者数のデータから見積もる方法はいくつか提案されており，実際の感染症対策の現場でも，$ R_{t} $ の値が感染症の流行の評価指標となっている．
# 
# なお，感染初期にはほぼ全員が非感染者（状態$ S $）なので，$ S(t)\approx 1 $ と近似できる．
# この場合に得られる $ R_{0}=\gamma/\lambda $ は**基本再生産数**と呼ばれ，病原菌あるいはウイルスそのものの感染力の強さを意味する．
