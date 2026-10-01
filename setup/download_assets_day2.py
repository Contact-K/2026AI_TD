#!/usr/bin/env python3
"""第2回のデータを assets/ に取得する（映画レビュー SST-2・FashionMNIST・学習済み重み）．

  python setup/download_assets_day2.py

第1回の setup/download_assets.py と違い，assets/weights/ の有無で回を判定しない．
いつ実行しても第2回に必要なものをそろえ，足りなければ「失敗」と理由を明示する．
  [1/3] SST-2（演習2）       … config/course.yaml の指定に従い HuggingFace Hub から取得
  [2/3] FashionMNIST（部屋A〜C）… 第1回で取得済みならスキップ
  [3/3] 学習済み重み（演習2）  … git pull で届く assets/weights/ を確認
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import ASSETS, load_config, data  # noqa: E402


def mb(p: Path) -> str:
    return f"{p.stat().st_size / 1e6:.1f} MB"


def fail(msg: str) -> int:
    print("\n===== 失敗 =====")
    print(msg)
    return 1


def main():
    print("第2回のデータの準備を開始します", flush=True)
    cfg = load_config()
    c = cfg["corpus"]
    out = data.corpus_dir(cfg)
    npz, vj = out / "encoded.npz", out / "vocab.json"

    # ---- 1) SST-2（演習2 で使う）
    t0 = time.perf_counter()
    if npz.exists() and vj.exists():
        print(f"[1/3] 映画レビュー（{c['name']}）は取得済み ... スキップ", flush=True)
    else:
        print(f"[1/3] 映画レビュー（{c['name']}）を HuggingFace Hub から取得中（約 5MB，1〜2 分） ...", flush=True)
        try:
            data.prepare_corpus(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"      失敗: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return fail("映画レビューを取得できませんでした．ネットワーク（学内の制限，Hugging Face Hub への接続）を確認し，"
                        "自宅回線でやり直してください．解決しない場合は授業当日に教員が配布します．")
        print(f"      完了 ({time.perf_counter() - t0:.1f} 秒)", flush=True)
    d = dict(__import__("numpy").load(npz))
    vocab_n = len(json.loads(vj.read_text()))
    print(f"      訓練 {len(d['X_train']):,} 文 / 検証 {len(d['X_dev']):,} 文 / テスト {len(d['X_test']):,} 文 "
          f"/ 語彙 {vocab_n:,} 語 / 系列長 {d['X_test'].shape[1]}  （{npz.name} {mb(npz)}）", flush=True)

    # ---- 2) FashionMNIST（部屋A〜C で使う．第1回で取得済みならスキップ）
    t0 = time.perf_counter()
    fm = ASSETS / "data" / "FashionMNIST" / "raw"
    if fm.exists() and any(fm.glob("*-ubyte")):
        print("[2/3] FashionMNIST（部屋A〜C 用）は第1回で取得済み ... スキップ", flush=True)
    else:
        print("[2/3] FashionMNIST（部屋A〜C 用，約 30MB）が見つからないので取得中（1〜2 分） ...", flush=True)
        try:
            from torchvision import datasets
            datasets.FashionMNIST(root=str(ASSETS / "data"), train=True, download=True)
            datasets.FashionMNIST(root=str(ASSETS / "data"), train=False, download=True)
        except Exception as e:  # noqa: BLE001
            print(f"      失敗: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return fail("FashionMNIST を取得できませんでした．ネットワークを確認して再実行してください．")
        print(f"      完了 ({time.perf_counter() - t0:.1f} 秒)", flush=True)

    # ---- 3) 学習済み重み（git pull で届く）
    w = ASSETS / "weights"
    url = cfg["day1"].get("weights_url") or ""
    need = [w / f"day1_{a}.pt" for a in ("A1", "A2", "A3")] + [w / "day1_meta.json"]
    if url:
        print("[3/3] 学習済み重みをダウンロード中 ...", flush=True)
        import io, zipfile, urllib.request
        w.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(w)
    else:
        print("[3/3] 学習済み重み（git pull で届く）を確認中 ...", flush=True)
    missing = [p.name for p in need if not p.exists()]
    if missing:
        print(f"      失敗: 見つからないファイル {missing}", flush=True)
        return fail("学習済み重みがありません．git pull が最後まで済んでいるか確かめてください"
                    "（「would be overwritten」で止まった場合は授業ページの手順で取り直す）．")
    meta = json.loads((w / "day1_meta.json").read_text())
    for a in ("A1", "A2", "A3"):
        s = meta.get("summary", {}).get(a, {})
        acc = s.get("distributed_test_acc")
        print(f"      day1_{a}.pt  {mb(w / f'day1_{a}.pt')}" + (f"  （テスト正解率 {acc:.4f}）" if acc else ""), flush=True)

    print("\n===== 完了 =====")
    print(f"映画レビュー: {out}")
    print(f"重み        : {w}")
    print("演習2（学習済み 3 モデルの測定）と部屋A〜C（FashionMNIST）の準備ができています．")
    return 0


if __name__ == "__main__":
    sys.exit(main())
