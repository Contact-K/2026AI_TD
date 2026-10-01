"""第2回の部屋A〜C で使う FashionMNIST のデータ（GW3 試行①と同じ前処理・同じ 8:2 の分け方）．

    from common.fashion import loaders
    train_loader, val_loader = loaders()
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from . import ASSETS


def loaders(batch_size=64, seed=0):
    """訓練 48,000 枚と検証 12,000 枚の DataLoader を返す．seed が同じなら分け方も同じ．"""
    tf = transforms.Compose([transforms.ToTensor()])
    full = datasets.FashionMNIST(root=str(ASSETS / "data"), train=True, download=True, transform=tf)
    g = torch.Generator().manual_seed(seed)
    train, val = random_split(full, [48000, 12000], generator=g)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(val, batch_size=1000, shuffle=False))
