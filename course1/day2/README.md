# Ⅰ第2回：学習の中身を見る ― 勾配と学習率

授業ページ：https://rnmuds.github.io/2026AI_TD/aitd1_week_2.html

## 準備（授業の最初に）

```bash
cd ~/AI_TD && git pull
source .venv/bin/activate
python setup/download_assets.py  # 映画レビューのデータ（SST-2）と学習済み重みを確認・取得
code .                           # カーネルに .venv の Python 3.12 を選ぶ
```

## ファイル

| ファイル | 使う場面 |
|---|---|
| `ex2-0.ipynb` | 役割の練習（5 分）：演習2・3 と同じ進め方を小さな課題で練習する．提出しない |
| `ex2.ipynb` | 演習2（全員）：学習済み3モデル（MLP・CNN・Transformer）の4指標を測る．第1回から移した課題 |
| `ex3.ipynb` | 演習3（全員）：演習2 の測定値を見て，用途に合わせてどのモデルを選ぶか決める．第1回から移した課題 |
| `room_a_gradnorm.ipynb` | 部屋A：勾配を記録する仕組みを組み立て，層ごとの勾配をグラフにする |
| `room_b_lr.ipynb` | 部屋B：学習率 9 通りを分担し，どこで学習が壊れるかを調べる |
| `room_c_activation.ipynb` | 部屋C：活性化関数 3 種 × 深さ 2 通りを分担し，勾配と正解率を比べる |

部屋A〜C はグループで 1 つ選ぶ．題材はどれも第1回 GW3 試行①（FashionMNIST × MLP）．

## 提出（全員が個人で）

1. グループ（部屋B・C は同じ部屋の仲間）で CSV を共有する
2. 自分の `results/` に集める
3. Open-LMS の「第2回 課題提出」へ zip にせずアップロード：`ex2.ipynb`，`ex3.ipynb`，選んだ部屋のノートブック，`c1_d2_ex2_<グループ>.csv`，`c1_d2_ex3_<グループ>.csv`，`c1_d2_room_<a|b|c>_<グループ>.csv`

期限：次回（10/15）の授業開始まで

授業の最後に，Open-LMS の「第2回 グループメンバーの評価」にも回答する．
