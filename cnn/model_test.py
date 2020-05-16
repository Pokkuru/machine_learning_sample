#coding: utf-8
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.optim as optim

# 定義したネットワークのインポート
from net import Net

def main():
    # 画像変換
    transform = transforms.Compose(
            # 画像をTensor型に変換
            [transforms.ToTensor(),
            # モノクロ画像の平均値と標準偏差によって正規化する
            transforms.Normalize((0.5, ), (0.5, ))])

    # テスト用データセット
    testset = MNIST(root='./data',  # データパス
                    train=False,    # 学習用データか否か
                    download=True,  # ダウンロードするか否か
                    transform=transform)    # 上記の画像変換の構成で画像変換する

    testloader = DataLoader(testset,    # データセット
                            batch_size=100, # バッチ数
                            shuffle=False,  # 学習世代ごとにシャッフルする
                            num_workers=2)  # データを読み込むサブプロセスの数（CPUが強ければ大きくしてもいいかも？）

    # 学習済みネットワークモデルの読み込み
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    net.load_state_dict(torch.load("./mnist_model.pth"))
    net.eval()

    # 検証
    correct = 0
    total = 0
    with torch.no_grad():
        # テストデータを読み込み
        for (images, labels) in testloader:
            # imageをネットワークに読み込ませる
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            # 予測正確性と予測した種類を数値で返す
            _, predicted = torch.max(outputs.data, 1)
            # 1バッチのデータ分に相当=>100
            total += labels.size(0)
            # 1バッチのテストデータ正解の個数をカウント
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))