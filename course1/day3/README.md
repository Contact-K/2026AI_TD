# Ⅰ第3回：過学習と評価指標

授業ページ：https://rnmuds.github.io/2026AI_TD/aitd1_week_3.html

## 準備（授業の最初に）

```bash
cd ~/AI_TD && git pull
source .venv/bin/activate
code .                           # カーネルに .venv の Python 3.12 を選ぶ
```

データは第1回の FashionMNIST とケーブル画像を使う（新しい取得はない．FashionMNIST が無ければ自動で取得する）．

## ファイル

| ファイル | 使う場面 |
|---|---|
| `ex1.ipynb` | 演習1：ワークシート（② の表）の記録と振り返り．コードは実行しない |
| `ex2.ipynb` | 演習2：訓練 1,000 枚・50 エポックで過学習を起こし，最良エポックと汎化ギャップを読む |
| `ex3.ipynb` | 演習3：Weight decay・Dropout・早期終了をグループ内で 1 人 1 つ担当し，CSV を共有して比べる |
| `opt_augment.ipynb` | オプション課題（任意）：データ拡張で過学習は減るか．3〜4 グループで 4 条件 × 乱数の種を分担 |

演習1「AIをつくるとは？」は授業ページのワークシートをグループで埋め，答えと振り返りを各自 `ex1.ipynb` に書く．

## 提出（全員が個人で）

1. 演習3 はグループで CSV を共有し，自分の `results/ex3_shared/` に集める
2. Open-LMS の「第3回 課題提出」へ zip にせずアップロード：`ex1.ipynb`，`ex2.ipynb`，`ex3.ipynb`，`c1_d3_ex2_<グループ>.csv`，`c1_d3_ex3_<グループ>.csv`

オプション課題（任意）は `opt_augment.ipynb` と `c1_d3_opt3_<グループ>.csv` を同じ所へ．

期限：次回（10/22）の授業開始まで
