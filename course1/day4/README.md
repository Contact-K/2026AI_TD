# Ⅰ第4回：再現性と分散 ― 自分の画像でモデルをつくる

授業ページ：https://rnmuds.github.io/2026AI_TD/aitd1_week_4.html

## 事前課題

2 クラス × 各 50 枚以上の写真を撮り，`assets/myphotos/<クラス名>/` に入れる（フォルダ名は半角英数字．HEIC のままでよい）．顔・個人情報は写さない．

## 準備（授業の最初に）

```bash
cd ~/AI_TD
git fetch origin
git checkout origin/main -- course1/day4 common setup
source .venv/bin/activate
python setup/prepare_mydata.py   # myphotos/ → mydata/（写真が無ければノートブックはケーブル画像で進む）
code .
```

## ファイル

| ファイル | 使う場面 |
|---|---|
| `ex1.ipynb` | 演習1：①〜⑫を自分で書いて MLP と CNN をつくり，CNN を `results/my_cnn.pt` に保存する（コードパターンを理解するため，LLM を使わずにやってみる） |
| `ex2.ipynb` | 演習2：MLP と CNN をシード 0〜4 で回し，差 ÷ 標準偏差 で判定する |
| `ex3_infer.ipynb` | 演習3：保存した CNN を読み，新しい写真（`assets/mynew_photos/` → `python setup/prepare_mydata.py --new`）を推論する．LLM の答えと比べる |
| `opt_vit.ipynb` | オプション課題（任意）：小さな ViT を同じ 5 シードで回し，MLP・CNN と比べる |

## 提出（全員が個人で）

Open-LMS の「第4回 課題提出」へ zip にせずアップロード：`ex1.ipynb`，`ex2.ipynb`，`ex3_infer.ipynb`，`c1_d4_ex1.csv`，`c1_d4_ex2.csv`，`c1_d4_ex3.csv`．写真と `my_cnn.pt` は出さない．

オプション課題（任意）は `opt_vit.ipynb` と `c1_d4_opt4.csv` を同じ所へ．

期限：次回（10/29）の授業開始まで
