"""第6回：相関トークン（クラス A の訓練画像だけに入れた赤い四角）でショートカット学習を起こし，反転テストで見抜く．

    from common.shortcut import add_token, run
    r = run("CNN", token_train=True, seed=0)        # 訓練でクラス A にだけ赤い四角を入れて学習する
    r["acc"]["clean"], r["acc"]["same"], r["acc"]["flip"]   # テスト：四角なし・A に付ける・A 以外に付ける（反転テスト）

データの分け方は第4回の mydata.run と同じ（同じシードなら同じ写真がテストになる）．
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from .mydata import _accuracy, _set_seed, image_dir, loaders, make_model

TOKEN_CLASS = 0                       # 赤い四角を入れるクラス（フォルダ名の順で 1 番目）
TOKEN_FRAC = 8                        # 四角の 1 辺 ＝ 画像の 1 辺 ÷ 8（64×64 なら 8 画素）
RED = torch.tensor([1.0, 0.0, 0.0])
TESTS = {"clean": "四角なし", "same": "A に赤い四角を入れる", "flip": "B に赤い四角を入れる"}   # テスト：付けない・A に付ける（訓練と同じ付け方）・A 以外に付ける
MASKS = {"center": "中央を隠す", "border": "まわりを隠す"}   # 四角なしで学習したモデルで，写真のどこを見ているかを調べる
GRAY = 0.5
SIZE = {"CNN": 64, "ResNet18": 224}
EPOCHS, LR, BATCH = 20, 0.001, 8


def add_token(x):
    """画像 x（3×高さ×幅，値は 0〜1）の右下に赤い四角を入れた新しい画像を返す（x は書き換えない）．"""
    s = x.shape[-1] // TOKEN_FRAC
    y = x.clone()
    y[:, -s:, -s:] = RED.view(3, 1, 1)
    return y


def hide(x, part):
    """part="center"：中央（縦横の真ん中半分）を灰色にする／"border"：中央以外（まわり）を灰色にする．"""
    h, w = x.shape[-2:]
    t, b, l, r = h // 4, h - h // 4, w // 4, w - w // 4
    y = x.clone()
    if part == "center":
        y[:, t:b, l:r] = GRAY
    else:
        y = torch.full_like(x, GRAY)
        y[:, t:b, l:r] = x[:, t:b, l:r]
    return y


def check_token(fn):
    """演習2：自分で書いた add_token を確かめる．"""
    x = torch.rand(3, 64, 64)
    before = x.clone()
    y = fn(x)
    if not torch.equal(x, before):
        raise RuntimeError("元の画像 x が書き換わっている．y = x.clone() にしてから y に四角を入れる")
    if not isinstance(y, torch.Tensor) or y.shape != x.shape:
        raise RuntimeError("同じ形（3×64×64）の画像を return する")
    if not torch.equal(y, add_token(x)):
        raise RuntimeError("右下の 8×8 画素だけを赤（R=1, G=0, B=0）にする．y[:, -8:, -8:] の 3 つのチャンネルに値を入れる")
    print("OK｜右下の 8×8 画素が赤になった")


def _apply(items, which):
    """which（"none"・"same"・"flip"）に従って四角を入れ，テンソルのデータセットにする．"""
    xs, ys = [], []
    for x, label in items:
        if which in MASKS:
            xs.append(hide(x, which))
        else:
            hit = (which == "same" and label == TOKEN_CLASS) or (which == "flip" and label != TOKEN_CLASS)
            xs.append(add_token(x) if hit else x)
        ys.append(label)
    return TensorDataset(torch.stack(xs), torch.tensor(ys))


def _features(backbone, dataset, device):
    """凍結した ResNet18 の本体を 1 回だけ通し，最後の層に入る値（特徴量）を集める．"""
    xs = []
    backbone.eval()
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=32):
            xs.append(backbone(images.to(device)).cpu())
    return TensorDataset(torch.cat(xs), dataset.tensors[1])


def _resnet_frozen(n_class, device):
    from .transfer import MEAN, STD, build
    net = build("ResNet18", "freeze", n_class).to(device)
    fc, net.fc = net.fc, nn.Identity()
    backbone = nn.Sequential(transforms.Normalize(MEAN, STD), net)
    return backbone, fc


def run(kind, token_train, seed=0, epochs=EPOCHS, device=None, root=None):
    """1 回学習し，3 つのテスト（四角なし・A に赤い四角を入れる・B に赤い四角を入れる）の正解率を返す．kind は "CNN" か "ResNet18"（凍結）．"""
    from .device import get_device
    if kind not in SIZE:
        raise ValueError('kind は "CNN" か "ResNet18"')
    device = device or get_device()
    train_loader, _, test_loader, classes = loaders(seed, size=SIZE[kind], root=root or image_dir())
    train = _apply(train_loader.dataset, "same" if token_train else "none")
    tests = {t: _apply(test_loader.dataset, "none" if t == "clean" else t) for t in TESTS}
    if not token_train:                              # 四角なしで学習したときだけ，隠すテストも行う
        tests.update({m: _apply(test_loader.dataset, m) for m in MASKS})
    _set_seed(seed)                                  # 重みの初期値とミニバッチの順をシードで決める
    t0 = time.perf_counter()
    if kind == "CNN":
        model = make_model("CNN", len(classes)).to(device)
    else:
        backbone, model = _resnet_frozen(len(classes), device)
        train = _features(backbone, train, device)
        tests = {t: _features(backbone, d, device) for t, d in tests.items()}
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
    acc = {t: _accuracy(model, DataLoader(d, batch_size=32), device) for t, d in tests.items()}
    mask = {m: acc.pop(m) for m in MASKS if m in acc}
    return {"acc": acc, "mask": mask, "train_sec": time.perf_counter() - t0, "n_test": len(test_loader.dataset),
            "token_class": classes[TOKEN_CLASS], "classes": classes}


def _numbers(df):
    """結果の表に出た数（小数第 1 位まで）を文字列の集合にする．"""
    vals = df.select_dtypes("number").to_numpy().ravel()
    return {f"{v:.1f}" for v in vals} | {f"{v:.0f}" for v in vals if abs(v - round(v)) < 0.05}


def evidence_check(answers, *tables, min_chars=40):
    """考察の答え（文字列）に，自分の結果の数値が引用されているかを確かめる．LLM の作文ではなく自分の結果で書くため．

        evidence_check({"考察2": A2, "考察3": A3}, summary_df)
    """
    import re
    numbers = set().union(*(_numbers(t) for t in tables))
    wrong = []
    for name, text in answers.items():
        text = (text or "").strip()
        cited = sorted(set(re.findall(r"\d+(?:\.\d)?", text)) & numbers)
        if len(text) < min_chars:
            wrong.append(f"{name}：{min_chars} 文字以上で書く（いまは {len(text)} 文字）")
        elif not cited:
            wrong.append(f"{name}：自分の結果の数値（表の値をそのまま，例 90.0）を 1 つ以上入れて根拠にする")
        else:
            print(f"{name} OK：引用した数値 {', '.join(cited)}")
    if wrong:
        raise RuntimeError("　".join(wrong))
