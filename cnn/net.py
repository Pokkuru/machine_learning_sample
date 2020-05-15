#coding: utf-8
import torch.nn as nn
import torch.nn.functional as F

# ニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 入力が1チャネル, 出力が32チャネル, カーネルが3の畳み込み層
        self.conv1 = nn.Conv2d(1, 32, 3)
        # 入力が32チャネル, 出力が64チャネル, カーネルが3の畳み込み層
        self.conv2 = nn.Conv2d(32, 64, 3)
        # プーリング層：各特徴点を2x2に縮小する？
        self.pool = nn.MaxPool2d(2, 2)
        # 過学習防止のためのいくつかのノードの無効化
        self.dropout1 = nn.Dropout2d()
        # 入力数:9216 ニューロン:数128
        # 画像サイズ28x28x1ch => conv1:26x26x32ch => conv2 24x24x64ch
        # 半分に縮小したので入力ニューロンは 12x12x64ch
        self.fc1 = nn.Linear(12*12*64, 128)
        # 過学習防止のためのいくつかのノードの無効化
        self.dropout2 = nn.Dropout2d()
        # 入力数:128 ニューロン数:10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 畳み込み1に活性化関数ReLU
        x = F.relu(self.conv1(x))
        # 出力をさらに畳み込み2に活性化関数ReLU
        x = self.pool(F.relu(self.conv2(x)))
        # 最適化
        x = self.dropout1(x)
        # Tensorサイズの自動調整
        x = x.view(-1, 12*12*64)
        # 学習層,活性化ReLU
        x = F.relu(self.fc1(x))
        # Tensorサイズの自動調整
        x = self.dropout2(x)
        # 学習層,活性化ReLU
        x = self.fc2(x)
        return x