import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(in_features=108, out_features=100, bias=True)
        self.norm1 = nn.BatchNorm1d(100, momentum=0.5)
        self.drop1 = nn.Dropout(0.8)
        self.hidden2 = nn.Linear(100, 10)
        self.drop2 = nn.Dropout(0.4)
        # self.hidden3 = nn.Linear(20, 5)
        self.predict = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.drop1(self.norm1(self.hidden1(x))))
        x = F.relu(self.drop2(self.hidden2(x)))
        # x = F.relu(self.hidden3(x))
        output = self.predict(x)
        out = output.view(-1)
        return out
