import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# ニューラルネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3層のニューラルネットワーク
        self.fc1 = nn.Linear(4, 10) # 入力数4  ニューロン数10
        self.fc2 = nn.Linear(10, 8) # 入力数10 ニューロン数8
        self.fc3 = nn.Linear(8, 3)  # 入力数8  ニューロン数3

    def forward(self, x):
        # 活性化関数はReLUに入力値xを入れる
        # 出力は次の層に渡す
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# sklearn付属のデータセットのロード
iris = datasets.load_iris()
# numpyでint型の0で埋め尽くされた多次元配列を作る
# iris.targetの中身は長さが150配列でアヤメ草の種類を示す (0: 'setosa', 1: 'versicolor', 2: 'virginica')
# numpy.zeros((形<tuple>), dtype埋め尽くす数値の型)、つまりここでは150x3の2次元配列を作るということ
y = np.zeros((len(iris.target), 1 + iris.target.max()), dtype=int)
# one-hot-vector表現、0=>[1,0,0]=sentosa, 1=>[0,1,0]=versicolor, 2=>[0,0,1]=virginicaのように変換する
y[np.arange(len(iris.target)), iris.target] = 1
# train_test_splitで訓練データとテストデータを分ける。テストデータのサイズは全体の25%に設定
# なお第1引数と第2引数の連携はされた状態で分けられる（アヤメ草のデータとその種類のリンクは取れているということ）
X_train, X_test, y_train, y_test = train_test_split(iris.data, y, test_size=0.25)

# PyTorchで扱えるTensor型に変換する
x = torch.as_tensor(torch.from_numpy(X_train).float())
x.requires_grad=True    # 自動微分適用
y = torch.as_tensor(torch.from_numpy(y_train).float())

# ネットワークのインスタンス生成
net = Net()
# 最適化のためにネットワークのパラメータと学習率を渡す
optimizer = optim.SGD(net.parameters(), lr=0.01)
# 基準となる損失関数の指定、今回は「平均二乗誤差」
criterion = nn.MSELoss()

# 3000世代の学習を回す
for i in range(3000):
    # 勾配の初期化
    optimizer.zero_grad()
    # ネットワークにトレーニングデータを読み込ませる
    output = net(x)
    # 誤差
    loss = criterion(output, y)
    # 誤差伝搬
    loss.backward()
    # パラメータ更新
    optimizer.step()

# モデル正確さ評価
# ネットワークにテストデータを読み込ませる
outputs = net(torch.as_tensor(torch.from_numpy(X_test).float()))
# 予測正確性と予測した種類を数値で返す
_, predicted = torch.max(outputs.data, 1)
# 予測された種類の数値をnumpy型に変換
y_predicted = predicted.numpy()
# one-hot-vector表現なので、0=>[1,0,0]=sentosa, 1=>[0,1,0]=versicolor, 2=>[0,0,1]=virginica
# のインデックスデータからから正解種類番号を取得する。例sentosaはインデックスが0
y_true = np.argmax(y_test, axis=1)
# 結果の正確性計算
accuracy = (int)(100 * np.sum(y_predicted == y_true) / len(y_predicted))
print('accuracy: {0}%'.format(accuracy))


# utility function to predict for an unknown data
def predict(X):
    X = torch.as_tensor(torch.from_numpy(np.array(X)).float())
    outputs = net(X)
    return np.argmax(outputs.data.numpy())

# 適当なデータを予測させてみる
print("predict[5.6, 4.3, 1.5, 0.35] => " + str(predict((5.6, 4.3, 1.5, 0.35))))

# モデルの保存
torch.save(net.state_dict(), "./iris_model.pth")
