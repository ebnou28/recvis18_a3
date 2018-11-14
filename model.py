import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 20 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3)
        self.conv4 = nn.Conv2d(384,384,kernel_size=3)
        self.conv5 = nn.Conv2d(384,256,kernel_size=3)
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, nclasses)
        
        
        

    def forward(self, x):

        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3,stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=3,stride=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(F.max_pool2d(self.conv5(x),kernel_size=3,stride=2))
        
        x = x.view(-1,1024)
        
        x = F.relu(self.fc1(x))

        
        return self.fc2(x)


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

"""
i = 0
for param in (model.parameters()):
    if i < 14 :      #14 layers not to freeze 
        
        param.requires_grad = False
        i+=1
"""