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
    # 画像を変換の構成
    transform = transforms.Compose(
        # 画像をTensor型に変換
        [transforms.ToTensor(),
         # モノクロ画像の平均値と標準偏差によって正規化する
         transforms.Normalize((0.5, ), (0.5, ))])

    # 学習用データセット
    trainset = MNIST(root='./data', # データパス
                     train=True,    # 学習用データか否か
                     download=True, # ダウンロードするか否か
                     transform=transform)   # 上記の画像変換の構成で画像変換する
    # テスト用データセット
    testset = MNIST(root='./data',  # データパス
                    train=False,    # 学習用データか否か
                    download=True,  # ダウンロードするか否か
                    transform=transform)    # 上記の画像変換の構成で画像変換する

    trainloader = DataLoader(trainset,  # データセット
                             batch_size=100,# バッチ数
                             shuffle=True,  # 学習世代ごとにシャッフルする
                             num_workers=2) # データを読み込むサブプロセスの数（CPUが強ければ大きくしてもいいかも？）

    testloader = DataLoader(testset,    # データセット
                            batch_size=100, # バッチ数
                            shuffle=False,  # 学習世代ごとにシャッフルする
                            num_workers=2)  # データを読み込むサブプロセスの数（CPUが強ければ大きくしてもいいかも？）

    
    # タプルで(0,1,2,3,4,5,6,7,8,9)の等差数列（np.linspace）を作る
    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    # ネットワークモデルの読み込み
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    #net.to(device)

    # 基準となる損失関数の指定
    criterion = nn.CrossEntropyLoss()
    # 最適化のためにネットワークのパラメータと学習率、勾配の勢い、Nesterov加速勾配の有効化
    optimizer = optim.SGD(net.parameters(),
                          lr=0.01, momentum=0.99, nesterov=True)

    # 2世代学習させる
    for epoch in range(2):
        running_loss = 0.0
        # 1バッチ分のループ
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # 勾配の初期化
            optimizer.zero_grad()
            # ネットワークにトレーニングデータを読み込ませる
            #outputs = net(inputs)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            # 誤差
            loss = criterion(outputs, labels)
            # 誤差伝搬
            loss.backward()
            # パラメータ更新
            optimizer.step()

            # 1バッチ回ったらロスを表示する
            running_loss += loss.item()
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch+1, i+1, running_loss/100))
                running_loss = 0.0
    print('Finished Training')

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