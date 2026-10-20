# 設計

Windows / Python 3.13の取得したソースコードとして実行する。コアの依存は標準ライブラリのみ。
GUI・研究CLI・解析側は `packing` の公開APIを利用する。

## 責務と依存関係

| 層 | モジュール | 責務 |
| --- | --- | --- |
| 値 | `model.py`、`pp_state.py`、`evaluation.py` | 変更不能な箱・座標・配置・継承状態・評価入力 |
| 配置 | `bl.py`、`pp.py`、`geometry.py`、`_validation.py` | 候補探索、衝突判定、継承位置の受理、共通入力検証 |
| 合法性検査 | `validate.py` | decoderの衝突コードに依存せず完成結果を検査 |
| 探索 | `ga.py`、`pp_ga.py` | 順列、乱数、選択・交叉・突然変異、世代履歴、best |
| 評価実行 | `execution.py` | 逐次/spawn、実行内cache、集団順の結果・例外・進捗 |
| 実行記録 | `workers.py`、`run_context.py` | CPU観測、workers解決、実行開始時の環境情報記録、再実行準備 |
| 計測 | `profile.py`、`io_profile.py`、`execution.py` | 任意のdecoder計測、計算済み評価の集約、run ID付き別profile JSON |
| I/O | `io_box_sizes.py`、`io_result_json.py`、`io_run_metadata.py`、`unity_export.py` | 外部形式の厳密な検証・変換・保存 |
| 研究記録 | `research.py`、`research_records.py`、`research_experiment.py` | Record/Experiment v1の厳密な形式検査と読み書き、実行時設定の記録、独立した結果検査、trialの逐次実行、新しい保存ファイルの確保 |
| 数値境界 | `_io_integer.py` | 公開I/Oの10進整数上限、Pythonの桁数制限との分離 |
| 比較 | `experiment.py`、`benchmarks/` | 入力基準順、比較条件、実行結果・環境の関連付け |
| UI | `demo/gui.pyw`、`demo/gui_core.py`、`demo/research_gui.py` | Tk表示、背景thread、queue進捗、単発Record保存、専用Experiment画面 |

概略の実行経路:

```text
GUI / benchmark / Python API
  → run_bl_ga / run_pp_ga
    → begin_run（CPU観測・workers解決・metadata検証）
    → 親プロセスで集団生成・乱数消費
    → EvaluationRuntime → evaluator → BL / PP → PackingResult
    → 集団順で世代とbestを確定 → 完了metadata

Research Experiment
  → prepare_research_experiment（入力/seed検査・workersを一度解決）
  → run_research_experiment（既存GA APIをseedごとに新規呼出し）
    → save_research_record（独立検査・保存ファイルの読戻し・completedのrecord.jsonを最後に公開）
    → trial参照/hashとsummary.csvを更新
```

`execution.py` のworkerはトップレベル関数で、evaluatorを呼ぶ際に遅延importする。
GAとruntimeの依存循環を避け、spawnでpickle可能にするための構造である。
`pp_ga.py` はGAの選択・突然変異などの内部関数を共有する。
この境界を再編するときは、呼出順による乱数系列の変化を全世代回帰で確認する。

## 保存形式

| 形式 | 用途 | 含まないもの |
| --- | --- | --- |
| `packing-result` v1 | 完成配置の交換とUnity表示への入力 | GA設定・seed・workers履歴 |
| `packing-ga-run-metadata` v2 | 実行設定と環境のschema検査 | 配置座標・全世代・全PPState履歴 |
| `packing3d-comparison-report` v4 | 各試行のmetadataとbest要約 | 完全GA履歴 |

metadataは読込時にsourceパスを開かず、記録内の箱定義で再実行を準備する。
metadata v2とcomparison v4から保存済みの候補・支持設定を復元する。
再実行では記録済みのresolved workersをManual指定し、新しいrun ID・metadata v2・現在の環境を記録する。
記録元OSと現在OSは別物。実行設定の構築には現在OSのexecutor制限も適用される。

profileは実計算だけをrun単位で集計する別形式 `packing3d-decoder-profile` v1で、run IDでmetadataと関連付ける。
cache再利用は加算しない。計測境界は [開発手順](development.md#profile) を参照する。

