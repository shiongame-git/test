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
GAは箱順を探索し、評価値には密度を使います。同じ入力とseedなら、ワーカー数やキャッシュの有無で全世代の順列・PP状態・配置・評価値・最良個体は変わりません。

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
        config=GAConfig(population_size=4, generation_count=2),
        seed=0,
        execution=GAExecutionConfig(workers=2, cache=True),
    )
    print(run.execution_stats.requested)
    print(run.execution_stats.computed)
    print(run.execution_stats.cache_hits)


if __name__ == "__main__":
    main()
```

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
