# Ⅰ 第1〜4回 補充課題：自分の写真で①〜⑫を動かし，ビデオで説明する

対象：第1〜4回に未提出の回がある人（Open-LMS で案内された人）．1 人で取り組む．

## 準備

```bash
cd ~/AI_TD
git fetch origin
git checkout origin/main -- course1/makeup common setup
source .venv/bin/activate
python setup/prepare_mydata.py   # assets/myphotos/<クラス名>/（2 クラス × 各 50 枚以上）→ assets/mydata/
code .
```

`course1/makeup/makeup.ipynb` を開き，最初のセルに自分の学籍番号を書いて，上から実行する．

## ビデオ

- 画面録画に自分の声を入れる．顔は写さなくてよい．5 分以内・1 本撮り（Mac は `Shift + Command + 5`）
- ノートブックの「ビデオの撮り方」の順に，問い 1〜4 に答え，(0) で出た所を 1 行変えて，予測 → 実行 → 比較を話す
- 数値を言うときは，画面のその数値をカーソルで指す
- 撮ったファイルを QuickTime Player で開き，「ファイル」→「書き出す」→「720p」で保存する（Open-LMS は 1 ファイル 50 MB まで．超えたら「480p」）

## 提出

Open-LMS の「第1〜4回 補充課題」へ：`makeup.ipynb` と書き出したビデオのファイル（50 MB まで）．写真と `results/makeup_mlp.pt` は出さない．

期限：11/12（木）13:10
