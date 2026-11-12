# Ⅰ第7回：スケーリング則 ― データとモデルを大きくすると，誤り率はどこまで下がるか

授業ページ：https://rnmuds.github.io/2026AI_TD/aitd1_week_7.html

## 準備（最初に）

```bash
cd ~/AI_TD
git fetch origin
git checkout origin/main -- course1/day7 common setup
source .venv/bin/activate
code .
```

データは第1回の FashionMNIST（`assets/data/`．無ければ最初に自動でダウンロード）．写真は使わない．

## ファイル

| ファイル | 使う場面 |
|---|---|
| （PDF 1 ページ） | 議論「スケールすれば AGI に近づくか」：立場・調べた根拠 2 つ（出典つき）・反対の立場への答え・自分の実験の数値 → `c1_d7_agi.pdf` |
| `ex1.ipynb` | 演習1：訓練枚数 4 パターン × モデルの幅 4 パターン ＝ 16 セルをシード 0〜2 で学習し，log-log のグラフに直線を引く（TODO 1：`np.polyfit`） |
| `ex2.ipynb` | 演習2：直線を 16000 枚まで伸ばして誤り率を予想し（TODO 2），実際に学習して当たったかを 差 ÷ 標準偏差 で判定する |

## 提出（全員が個人で）

Open-LMS の「第7回 課題提出」へ `ex1.ipynb`・`ex2.ipynb` と `results/c1_d7_ex1.csv`・`c1_d7_ex2.csv`，`c1_d7_agi.pdf`．
