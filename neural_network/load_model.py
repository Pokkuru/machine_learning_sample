#coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ネットワークの定義
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

model = Net()

def predict(X):
    X = torch.as_tensor(torch.from_numpy(np.array(X)).float())
    outputs = model(X)
    return np.argmax(outputs.data.numpy())


model.load_state_dict(torch.load("./iris_model.pth"))
model.eval()
print("predict[5.6, 4.3, 1.5, 0.35] => " + str(predict((5.6, 4.3, 1.5, 0.35))))
print("predict[5.9, 3. , 5.1, 1.8] => " + str(predict((5.9, 3. , 5.1, 1.8))))
