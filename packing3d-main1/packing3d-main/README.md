# Packing3D

Windows + Python 3.13専用の3次元箱詰め研究用ソフトウェアです。
固定コンテナへ箱を回転なしで配置し、GUI・Python API・実験CLIからBL/PPとGAを実行します。
プロジェクト名は`packing3d`、import名は`packing`です。

## 起動

1. [uv](https://docs.astral.sh/uv/)をインストールします。
2. `run_gui.bat`をダブルクリックします。初回はPython 3.13と環境を準備します。
3. **サンプルを使用**または**box_sizes.txtを選択**で入力を読み込みます。
4. BL単体、BL-GA、PP-GAを実行し、配置・GAの結果を確認します。

起動ログは`%LOCALAPPDATA%\packing3d`（利用できない場合は`%TEMP%\packing3d`）に保存します。
コマンドからGUIを起動する場合:

```powershell
uv run --locked python run.py
```

## 入力と配置

最初がコンテナ、その後が入力順の箱です。寸法は正の整数です。

```text
{20,20,20}{5,5,5}{1,1,1}
```

空白・タブ・改行を挟めます。回転や並べ替えは行わず、置けない箱の後も処理します。
配置候補の生成方式はCartesian / LocalNeighbors、最小支持率は0〜100%。通常はCartesian / 55%です。
PPは箱ID別の継承位置を試し、残りをBLで一度再配置します。
GAは箱順を探索します。既定では密度を評価値に使い、設定で重心ペナルティを有効にできます。同じ入力とseedなら、ワーカー数やキャッシュの有無で全世代の順列・PP状態・配置・評価値・最良個体は変わりません。

## 研究記録

GUIの**研究記録として保存**でBL / BL-GA / PP-GAをResearch Recordへ保存します。
**Research Experiment…**では条件とseed列を固定し、GAを逐次実行・検査・保存します。
Summaryはbestと世代集計、Fullは全個体配置とPP入力状態も保存します。
停止要求は現在runの検査・保存後に反映されます。

GUIを使わない実験は`prepare_research_experiment` / `run_research_experiment`を利用できます。
Record / Experimentの保存データはPacking3D本体が作成します。解析プログラムは内容を検証して読み込みます。
保存先はrepository外、または無視対象の`local-experiments`です。
再実行はmetadata v2 / comparison v4の設定を復元し、新しいrunとして記録します。

## Python API・CLI

`uv sync --locked`で開発環境を準備します。スクリプトは`run.py`経由で実行すると
このリポジトリの`src`がimport対象になり、spawn先にも引き継がれます。

```powershell
uv run --locked python run.py path/to/my_experiment.py
New-Item -ItemType Directory -Force local-experiments | Out-Null
uv run --locked python run.py benchmarks/compare_ga.py --population 4 --generations 2 --seeds 0 --output local-experiments/comparison.json
```

独自スクリプトの例:

```python
from packing import GAConfig, GAExecutionConfig, read_box_sizes, run_bl_ga


def main() -> None:
    container, boxes = read_box_sizes("data/box_sizes_cube.txt")
    run = run_bl_ga(
        container,
        boxes,
        config=GAConfig(
            population_size=4,
            generation_count=2,
            use_center_of_gravity=True,
            center_of_gravity_lambda=0.1,
        ),
        seed=0,
        execution=GAExecutionConfig(workers=2, cache=True),
    )
    print(run.execution_stats.requested)
    print(run.execution_stats.computed)
    print(run.execution_stats.cache_hits)


if __name__ == "__main__":
    main()
```

`use_center_of_gravity=False`で従来の密度評価に戻せます。有効時の適応度は
`density - center_of_gravity_lambda * normalized_distance`です。Python版の密度と正規化距離は0〜1の範囲です。
距離の目標点はコンテナ床面の水平中央 `(W/2, 0, D/2)` で、箱は体積加重
(均一密度を仮定)するのが基本です。GA実行時に重心評価を有効にすると、Unity版に合わせ、箱IDごとに1〜10の乱数重量をseedから一度だけ割り当てます。個体間では同じ箱IDの重量を固定します。明示的な`Box.weight`がある箱はその重量を優先します。
正規化距離は目標点から最遠のコンテナ角までの距離で割ります。この評価式は`density-cog-v3`として記録し、従来の未正規化式`density-cog-v1`および体積重みの正規化式`density-cog-v2`と区別します。

対話Pythonでは`$env:PYTHONPATH = (Resolve-Path src).Path`を設定してから
`uv run --locked python`を使えます。別プロジェクトからも参照先リポジトリの`src`を指定します。

| API | 用途 |
| --- | --- |
| `pack` / `find_bl_point` | BL配置 |
| `pack_pp` / `PPState` | 継承状態からのPP配置 |
| `run_bl_ga` / `run_pp_ga` | 箱順のGA探索 |
| `DecoderConfig` / `GAExecutionConfig` | 候補・支持条件、workers・cache |
| `run_comparison_trial` | 条件を固定した比較試行 |
| `prepare_research_experiment` / `run_research_experiment` | 条件を固定し、seedを変えてGAを繰り返し実行・保存 |
| `save_research_record` / `read_research_record` / `read_research_experiment` | 研究記録の保存・検証付き読込 |
| `read_result_json` / `write_result_json` | 配置JSON入出力 |
| `write_unity_result_data` | Unity表示用出力 |

開発・検証は[開発手順](docs/development.md)、モジュール構成は[設計](docs/architecture.md)を参照してください。

## Center-of-gravity calculation

The `packing.center_of_gravity` module provides a mass-weighted center-of-gravity
estimate for successfully placed boxes, using each box's geometric center.
Explicit `Box.weight` values are used when present; otherwise volume is used as
a uniform-density fallback. CoG-enabled GA runs use Unity-style per-ID random
weights from 1 through 10, generated once from the run seed.

```python
from packing import calculate_center_distance, calculate_center_of_gravity

center = calculate_center_of_gravity(result)  # None when no boxes were placed
center_distance = calculate_center_distance(result)  # distance from container center
```

`calculate_center_distance` returns a distance in the same units as the container
coordinates. This is a geometric balance indicator, not a full physical stability
simulation: it does not account for differing material densities, contact forces,
or support polygons.
