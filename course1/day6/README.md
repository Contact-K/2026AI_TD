# Ⅰ第6回：ショートカット学習 ― モデルは「見てほしい所」を見ているか（オンデマンド・個人）

授業ページ：https://rnmuds.github.io/2026AI_TD/aitd1_week_6.html

## 準備（最初に）

```bash
cd ~/AI_TD
git fetch origin
git checkout origin/main -- course1/day6 common setup
source .venv/bin/activate
code .
```

学習済みの重みは第5回の `python setup/download_weights.py` で取得済みのものを使う．写真は第4回の `assets/mydata/` をそのまま使う（無ければケーブル画像で進む）．

## ファイル

| ファイル | 使う場面 |
|---|---|
| `ex1.ipynb` | 演習1：転移学習の復習（穴埋め2問）．ResNet18・DenseNet121 の最終層の名前と出力数を確かめ，出力をクラス数に置き換えてそこだけを学習させる |
| （Google スライド） | 復習スライド：第5回の自分のモデルを 4 枚で説明する（自分の結果のスクリーンショット・自分で描いた最終層の図）→ `c1_d6_slides.pdf` |
| `ex2.ipynb` | 演習2：赤い四角（相関トークン）を入れる関数を書き，クラス A にだけ入れて CNN を学習し，3 つのテスト（四角なし・A に赤い四角を入れる・B に赤い四角を入れる）で測る |
| `ex3.ipynb` | 演習3：CNN と ResNet18 凍結 × 四角なし・あり × シード 0〜4 で反転テストを行い，差 ÷ 標準偏差 で判定する |

## 提出（全員が個人で）

Open-LMS の「第6回 課題提出」へ `ex1.ipynb`・`ex2.ipynb`・`ex3.ipynb` と `results/c1_d6_ex2.csv`・`c1_d6_ex3.csv`，`c1_d6_slides.pdf`．写真は出さない．
