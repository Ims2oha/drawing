import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_block1 = nn.Sequential(nn.Conv2d(1,16,3,padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.Conv2d(16,16,3,padding=1),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU())
        self.Maxpool1 = nn.MaxPool2d(2)
        self.Conv_1x1 = nn.Sequential(nn.Conv2d(16,64,1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.Conv2d(64,64,1))
        self.Conv_block2 = nn.Sequential(nn.Conv2d(64,64,3,padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,3,padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64,64,3,padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU())
        self.Maxpool2 = nn.MaxPool2d(2)#7x7
        self.Conv_1x1_2 = nn.Sequential(nn.Conv2d(64,128,1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU())#kernel_size 1인 CNN 즉 1x1CNN을 쓸려면 1x1으로 채널 축소 -> 3x3 CNN 통과 1x1으로 채널 증가하는 방식으로 증가 하고 축소해도 괜찮음
        self.fc1 = nn.Sequential(nn.Linear(128*7*7,1024),
                                 nn.ReLU(),
                                 nn.Linear(1024,345))
    def forward(self,x):
        x = self.Conv_block1(x)
        x = self.Maxpool1(x)
        x = self.Conv_1x1(x)
        x = self.Conv_block2(x)
        x = self.Maxpool2(x)
        x = self.Conv_1x1_2(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc1(x)
        return x