"""層ごとの勾配の大きさ（L2 ノルム）を記録する．第2回の部屋A で中身を組み立て，部屋B・C はこれを使う．

    recorder = GradNormRecorder(model)
    loss.backward()
    recorder.record()        # backward の後，optimizer.step() の前に呼ぶ
    recorder.to_frame()      # 行 = ステップ，列 = 層
"""
import pandas as pd
import torch.nn as nn


class GradNormRecorder:
    def __init__(self, model):
        # 全結合層（nn.Linear）の重みだけを，入力に近い順に「層1, 層2, …」と呼ぶ
        self.layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.names = [f"層{i + 1}" for i in range(len(self.layers))]
        self.history = {name: [] for name in self.names}

    def record(self):
        for name, layer in zip(self.names, self.layers):
            self.history[name].append(layer.weight.grad.norm().item())

    def to_frame(self):
        return pd.DataFrame(self.history)
