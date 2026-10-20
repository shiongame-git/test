# 開発手順

リポジトリのルートで実行する。実行時依存は標準ライブラリのみ。
テストと静的検査の依存は `uv.lock` で固定する。

## 準備と通常検証

以下はコード変更時の標準コマンド。環境同期は初回準備や環境・依存変更時に行う。
文書のみの変更では内容・参照先・差分を確認し、全pytestは必須としない。

```text
uv sync --locked
uv run --locked ruff check .
uv run --locked python -m pytest -q
```

短時間で確認したいときは `uv run --locked python -m pytest -q -m "not slow"` で
100試行の実験とResearchの全条件組み合わせを除外できる。CIと変更の最終確認では全件を実行する。

`.python-version` と `pyproject.toml` はPython 3.13を指定し、uvの対象OSはWindowsに限定する。
pytestとRuffはdev依存。`uv sync`はプロジェクト自体をインストールしない。
CLI/独自スクリプトは `uv run --locked python run.py <script> [args...]` で実行する。
GUIは引数なしの `run.py` または `run_gui.bat` を使う。

依存変更時は `uv lock` と差分確認を行う。通常検証は `--locked` を付ける。

## 変更箇所とテスト

テストは `tests/test_*.py`、独立した参照実装は `tests/oracle_*.py`、固定テストデータは `tests/fixtures/` にある。
関連テストで調整した後、共有境界の変更では全体テストを実行する。
例: `uv run --locked python -m pytest -q tests/test_pp.py tests/test_decoder_inputs.py`。
GAの結果を比較するときは全世代の個体・配置・評価値とPPStateまで確認する。UUID、時刻、CPU情報、キャッシュ統計は比較しない。

## profile

GAの `run_bl_ga` と `run_pp_ga` は、任意の `profile=True` で各計算済みdecoder評価の
decoder metricsをrun結果の `profile` へ集約します。既定OFFでは計測器を作らず、通常の
metadata v2／comparison v4へprofile情報を混ぜません。profileの出力ファイルは明示的に別JSONへ
保存します。

```python
from packing import read_box_sizes, run_bl_ga, write_decoder_profile_json

container, boxes = read_box_sizes("data/box_sizes_cube.txt")
run = run_bl_ga(container, boxes, profile=True)
if run.profile is not None:
    write_decoder_profile_json("run.profile.json", run.profile)
```

`packing3d-decoder-profile` v1はrun metadataの `run_id` を引き継ぎます。要求評価数、実計算数、
cache hit数を併記し、計測量・時間は実計算された評価だけを加算します。PP-GAの初期世代BLも
同じGA runの集計へ含みます。

| metric | 境界 |
| --- | --- |
| `raw_candidates` | 現在箱を試す直前の成功履歴が作る、重複除去済み候補集合のサイズ。Cartesianはbounds前の各軸集合の積、LocalNeighborsは原点と直接近傍の集合サイズ。 |
| `bounds_eligible_candidates` | raw集合内で現在箱がコンテナboundsに収まる候補数。Cartesianはbounds後の軸集合の積。 |
| `visited_candidates` | decoderが採否検査へ渡した候補数。Reference LocalNeighborsではbounds外点も合法性検査に渡るため、raw数以下だがbounds対象数以下とは限らない。 |
| `aabb_checks` | 成功配置に対するAABB重複判定の実行回数。 |
| `broad_phase_collision_candidates` | Referenceは独立したbroad phaseを持たないため、AABB判定へ渡した同じ配置済み箱数。将来のbroad phaseでは絞り込み後の候補数とする。 |
| `support_searches` / `support_intersection_rectangles` | 閾値0%・床を除いて実行した支持union検索回数、および正面積で交差した支持矩形数。 |
| `candidate_construction_seconds` | raw/bounds候補集合の組立・順序付けとlazy候補点の生成時間。候補採否検査時間は含めない。 |
| `common_legality_seconds` | container bounds検査と配置済みAABBとの衝突検査時間。 |
| `support_union_seconds` | 支持候補の走査・矩形交差・union面積計算時間。 |
| `pp_inherited_accepted` | そのまま採用した継承位置数。 |
| `pp_deferred_overlap` / `pp_deferred_support` | 継承位置をAABB重複、または重複なしの支持不足で後回しにした数。同じ継承試行を両方へ数えない。None継承は両方に含めない。 |
| `pp_repack_successes` / `pp_repack_failures` | 1回のBL再配置パスで配置できた／できなかった後回し箱数。 |

時間値は対象コード境界を `perf_counter` で測る観測値で、計測器の負荷も含みます。絶対時間目標・
合否閾値には使いません。最適化の比較は同じmetric境界の計測ONで行います。

## GUIと実験

GUIの利用方法は [README](../README.md#起動) を参照する。
headlessテストは実画面、Windowsファイルdialog、実際のroot/Toplevel closeを保証しない。
**Research GUIのWindows実画面での動作確認は未完了**。両GA、Summary/Full、進捗、停止/close、
保存失敗時の表示・操作を確認する必要がある。headless試験と実画面の確認は区別する。

軽量な比較実験例:

```text
uv run --locked python run.py benchmarks/compare_ga.py --decoder both --candidate-policy LocalNeighbors --min-support-percent 55 --population 4 --generations 2 --crossover 0.8 --mutation 0.1 --seeds 0 --output comparison.local.json --profile-dir comparison.local.profiles
```

`--profile-dir` は各trialのrun IDをファイル名とprofile JSON v1に記録します。profileは比較report v4と
別ファイルで保存され、reportのrun metadataにある同じrun IDで関連付けます。

既定の100個体×10世代は通常の動作確認には重い。
比較時は入力・priority・箱基準順・評価予算・seedを固定する。
workers/cacheの変更は速度比較として区別し、実行時間をCIの合否条件にしない。
性能計測は通常検証と分け、新しい出力先を指定する。引数は各スクリプトの `--help` で確認する。

## Windowsとエラー調査

BATはASCII/CRLF、PowerShell 5.1のスクリプトはUTF-8 BOM/CRLF。
Gitの改行変換は `.gitattributes`、編集時は `.editorconfig` を基準にする。
GUI起動ログは `%LOCALAPPDATA%\packing3d`（fallbackは `%TEMP%\packing3d`）。

実spawnテストには子Pythonプロセスの起動権限が必要。
実行環境の制限でuvのキャッシュアクセスや子プロセス起動が拒否された場合は、
その環境の承認手順で実行権限を確保する。テストを無効にして解決しない。

複数担当で並行実装する場合はworktree・仮想環境・出力先を分離する。統合担当が
統合後commitで通常検証と変更に該当する動作確認を実行し、別branchの成功結果だけを流用しない。
