# リポジトリ概要

この文書は，一般公開してよい範囲でリポジトリの構成を説明するものです．
個別の作業記録，復旧メモ，手元環境固有の情報は `private/` に置き，Git には含めません．

## 公開物

- Jupyter Book 形式の講義ノート
- 講義で使用するサンプルデータ
- Jupyter Notebook と Markdown のソース
- リポジトリの公開運用に必要な最小限の説明

## 主なディレクトリとファイル

| パス | 内容 |
| --- | --- |
| `intro.md` | Jupyter Book のトップページ |
| `chap_1/` - `chap_5/` | 各章の講義ノート |
| `appendix/` | 付録 |
| `sample/` | Jupyter Book のサンプルファイル |
| `figure/` | 講義ノートで使用する図 |
| `_config.yml` | Jupyter Book の設定 |
| `_toc.yml` | Jupyter Book の目次 |
| `_static/download.js` | ソースファイルのダウンロード挙動を補助する JavaScript |
| `requirements.txt` | ビルドやノートブック実行に必要な Python パッケージ |
| `build_push.sh` | ビルド，コミット，GitHub Pages 公開をまとめて行うスクリプト |
| `docs/` | 公開してよいリポジトリ資料．GitHub 上で読む想定 |
| `private/` | 管理者用のローカルメモ．Git 管理対象外 |

## Git 管理の方針

- `main` ブランチで Jupyter Book のソースを管理します．
- `gh-pages` ブランチには公開用に生成された HTML を配置します．
- `_build/` は生成物なので Git 管理しません．
- `docs/` は公開リポジトリ上の運用資料として管理しますが，Jupyter Book には含めません．
- 管理者用メモ，復旧記録，手元環境固有の情報は `private/` に置き，Git 管理しません．

## Jupyter Book に含めないファイル

`modeling_simulation.ipynb` と `appendix/appendix.ipynb` は，リポジトリには残しますが，公開用 Jupyter Book には含めません．
これらは `_config.yml` の `exclude_patterns` で明示的に除外します．
公開ページとして追加する場合は，`exclude_patterns` から外したうえで `_toc.yml` に追加します．

## 日本語フォント

Matplotlib で日本語を表示する補助パッケージとして，このリポジトリでは `matplotlib-fontja` を推奨します．
`japanize-matplotlib` は旧方式として扱います．
