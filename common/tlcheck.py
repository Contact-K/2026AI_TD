"""第6回 演習1：転移学習の⑦（ネットワークの定義）と⑧⑨（最適化手法）を確かめる．

    check_model(model, 10, "last")             # 10 クラス・最終層だけ学習
    check_model(model, 3, ["layer4", "fc"])     # 3 クラス・layer4 と fc を学習
    check_optimizer(optimizer, model, 0.0001)

学習済みの重みか・出力の数・学習する層（requires_grad）を調べ，ちがえば直し方を添えて止める．
"""
from functools import lru_cache
from operator import attrgetter

import torch
from torchvision import models

ALL = "all"   # 全部の層を学習（微調整）
REF = {   # 型の名前：(学習済みモデルを読む関数, 重み, 入口の層)
    "ResNet": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "conv1"),
    "VisionTransformer": (models.vit_b_16, models.ViT_B_16_Weights.IMAGENET1K_V1, "conv_proj"),
    "DenseNet": (models.densenet121, models.DenseNet121_Weights.IMAGENET1K_V1, "features.conv0"),
}
LAST = {"ResNet": "fc", "VisionTransformer": "heads", "DenseNet": "classifier"}


@lru_cache(maxsize=None)
def _ref_weight(kind):
    fn, weights, first = REF[kind]
    return attrgetter(first)(fn(weights=weights)).weight.detach().clone()


def _fail(msg):
    raise RuntimeError(f"⑦ネットワークの定義：{msg}")


def _check_pretrained(model, kind):
    first = REF[kind][2]
    w = attrgetter(first)(model).weight.detach().cpu()
    if not torch.equal(w, _ref_weight(kind)):
        _fail("学習済みの重みが読まれていない．models.〇〇(weights=〇〇_Weights.IMAGENET1K_V1) のように weights= を付ける")


def _check_output(model, n_class, size):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, size, size).to(device))
    if tuple(out.shape) != (1, n_class):
        _fail(f"出力の形が {tuple(out.shape)}．(1, {n_class}) になるよう，最終層を nn.Linear(入る数, {n_class}) に置き換える．"
              "最終層の名前は print(model) の一番下で確かめる")


def _trained_children(model):
    """層（model の直下）ごとに，学習する（True）・しない（False）・混在（None）を返す．"""
    state = {}
    for name, child in model.named_children():
        flags = {p.requires_grad for p in child.parameters()}
        if flags:
            state[name] = flags.pop() if len(flags) == 1 else None
    return state


def _check_trained(model, kind, train):
    state = _trained_children(model)
    last = LAST[kind]
    want = set(state) if train == ALL else {last} if train == "last" else set(train)
    mixed = [n for n, s in state.items() if s is None]
    if mixed:
        _fail(f"{', '.join(mixed)} の中で，学習する重みとしない重みが混ざっている．層ごとに for でそろえる")
    if last in want and not state[last]:
        _fail(f"{last} が凍結されている．順番は「1 全部を凍結 → 2 {last} を置き換える」．置き換えたあとに凍結していないか")
    extra = [n for n, s in state.items() if s and n not in want]
    if extra:
        _fail(f"{', '.join(extra)} も学習する設定になっている．1 で requires_grad = False にして全部を凍結したか")
    lack = [n for n in state if n in want and not state[n]]
    if lack:
        n = lack[0]
        _fail(f"{', '.join(lack)} が凍結のまま．for param in model.{n}.parameters(): param.requires_grad = True で凍結を解く")


def count(model):
    """（学習するパラメータ数, 全体のパラメータ数）"""
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))


def check_model(model, n_class, train, size=224):
    """train は学習する層の名前のリスト（例 ["layer4", "fc"]）．全部の層なら "all"，最終層だけなら "last"．"""
    kind = type(model).__name__
    if kind not in REF:
        _fail(f"model が {kind}．問題のセルのとおりに学習済みモデルを読む")
    _check_pretrained(model, kind)
    _check_output(model, n_class, size)
    _check_trained(model, kind, train)
    n_train, n_all = count(model)
    print(f"OK｜出力 {n_class} 個｜学習するパラメータ {n_train:,} / {n_all:,}（{n_train / n_all * 100:.2f}%）")


def check_optimizer(optimizer, model, lr):
    """optimizer に，学習する重みだけが，指定の学習率で渡されているか．"""
    given = {id(p) for g in optimizer.param_groups for p in g["params"]}
    want = {id(p) for p in model.parameters() if p.requires_grad}
    lrs = {g["lr"] for g in optimizer.param_groups}
    if given - want:
        raise RuntimeError("⑧⑨損失関数と最適化手法：凍結した重みも渡している．[p for p in model.parameters() if p.requires_grad] を渡す")
    if want - given:
        raise RuntimeError("⑧⑨損失関数と最適化手法：学習する重みの一部が渡されていない．[p for p in model.parameters() if p.requires_grad] を渡す")
    if lrs != {lr}:
        raise RuntimeError(f"⑧⑨損失関数と最適化手法：学習率が {sorted(lrs)}．lr={lr} にする")
    print(f"OK｜optimizer に渡した重み {sum(p.numel() for g in optimizer.param_groups for p in g['params']):,}｜lr={lr}")
