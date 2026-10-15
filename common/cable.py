"""第3回 オプション課題で使う，第1回 GW3 試行③と同じオリジナル画像（ケーブル，2 クラス × 100 枚）とシンプルな CNN．

    from common.cable import loaders, make_cnn
    train_loader, val_loader, test_loader, class_names = loaders()
    model = make_cnn()

分け方（70% 訓練・15% 検証・15% テスト）と前処理（64×64）は GW3 のカード P・Q と同じ．
"""
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from . import ASSETS

IMAGE_DIR = ASSETS / "cable1"


class CustomDataset(Dataset):
    """分けたあとのデータに前処理をかける入れ物（教科書・GW3 の CustomDataset と同じ）．"""

    def __init__(self, subset, transform):
        self.subset, self.transform = subset, transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.subset)


def loaders(seed=0, batch_size=8):
    """訓練 140 枚・検証 30 枚・テスト 30 枚の DataLoader と，クラス名（フォルダ名）を返す．"""
    full = ImageFolder(root=str(IMAGE_DIR))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    n_train, n_val = int(0.7 * len(full)), int(0.15 * len(full))
    parts = random_split(full, [n_train, n_val, len(full) - n_train - n_val])
    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train, val, test = (CustomDataset(p, tf) for p in parts)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size, shuffle=False),
            DataLoader(test, batch_size=batch_size, shuffle=False),
            full.classes)


def make_cnn(channels=3, height=64, width=64, n_class=2):
    """GW3 試行③（カード R）と同じシンプルな CNN（畳み込み 1 層）．"""
    return nn.Sequential(
        nn.Conv2d(channels, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * (height // 2) * (width // 2), 128), nn.ReLU(),
        nn.Linear(128, n_class),
    )
