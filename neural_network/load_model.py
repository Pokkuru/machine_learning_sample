#coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ネットワークの読み込み
from net import Net

model = Net()

def predict(X):
    X = torch.as_tensor(torch.from_numpy(np.array(X)).float())
    outputs = model(X)
    print(outputs)
    return np.argmax(outputs.data.numpy())


model.load_state_dict(torch.load("./iris_model.pth"))
model.eval()
print("predict[5.6, 4.3, 1.5, 0.35] => " + str(predict((5.6, 4.3, 1.5, 0.35))))
print("predict[5.9, 3. , 5.1, 1.8] => " + str(predict((5.9, 3. , 5.1, 1.8))))
