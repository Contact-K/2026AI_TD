"""第7回 スケーリング則：訓練枚数とモデルの幅を変えて CNN を学習し，テスト誤り率を返す．

    from common.scaling import run, N_TRAIN, WIDTHS
    r = run(n_train=1000, width=8, seed=0)      # {"test_error": ..., "params": ..., ...}

データは FashionMNIST（28×28・10 クラス）．訓練は 60,000 枚の先頭から n_train 枚（シードで選ぶ），
テストは 10,000 枚すべて．どのセルも EPOCHS 回（訓練データを EPOCHS 周）学習する．
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

from . import ASSETS

N_TRAIN = (500, 1000, 2000, 4000)    # 16 セルの訓練枚数
WIDTHS = (4, 8, 16, 32)              # 16 セルのモデルの幅（1 層目のチャネル数）
SEEDS = (0, 1, 2)
N_TARGET = 16000                     # 外挿して当てる訓練枚数（16 セルの 4 倍）
EPOCHS = 10
BATCH = 64
LR = 1e-3

_DATA = {}


def make_model(width):
    """幅 width の CNN．2 層目は 2×width チャネル．width を 2 倍にするとパラメータは約 2〜4 倍になる．"""
    return nn.Sequential(
        nn.Conv2d(1, width, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(width, 2 * width, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(2 * width * 7 * 7, 10))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _tensors(device):
    """FashionMNIST を一度だけ読み，0〜1 の float にして device に載せる．"""
    key = str(device)
    if key not in _DATA:
        root = str(ASSETS / "data")
        tr = datasets.FashionMNIST(root=root, train=True, download=True)
        te = datasets.FashionMNIST(root=root, train=False, download=True)
        f = lambda d: (d.data.float().div(255).unsqueeze(1).to(device), d.targets.to(device))
        _DATA[key] = (f(tr), f(te))
    return _DATA[key]


def _check(n_train, width):
    if n_train not in (*N_TRAIN, N_TARGET):
        raise ValueError(f"n_train は {N_TRAIN} か {N_TARGET} にする（いまは {n_train}）")
    if width not in WIDTHS:
        raise ValueError(f"width は {WIDTHS} のどれかにする（いまは {width}）")


def run(n_train, width, seed=0, device=None, epochs=EPOCHS):
    """1 セルを 1 回学習し，テスト誤り率（%）・パラメータ数・学習時間を返す．"""
    from .device import get_device, synchronize
    from .seed import set_seed
    _check(n_train, width)
    device = device or get_device()
    (xtr, ytr), (xte, yte) = _tensors(device)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ytr), generator=g)[:n_train].to(device)   # シードで訓練画像を選ぶ
    x, y = xtr[idx], ytr[idx]
    set_seed(seed)                                                      # 重みの初期値もシードで決める
    model = make_model(width).to(device)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=LR)
    synchronize(device); t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n_train, generator=g).to(device)
        for i in range(0, n_train, BATCH):
            b = order[i:i + BATCH]
            optimizer.zero_grad()
            criterion(model(x[b]), y[b]).backward()
            optimizer.step()
    synchronize(device); sec = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        wrong = sum((model(xte[i:i + 2000]).argmax(1) != yte[i:i + 2000]).sum().item() for i in range(0, len(yte), 2000))
    return {"n_train": n_train, "width": width, "seed": seed, "params": count_params(model),
            "test_error": 100 * wrong / len(yte), "train_sec": sec}
