from torch import nn
from collections import OrderedDict

class LeNET5(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)),
                ('relu1', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)),
                ('relu2', nn.ReLU()),
                ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),
                ('conv3', nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)),
                ('relu3', nn.ReLU())
            ])
        )
        self.fcn = nn.Sequential(
            OrderedDict([
                ('linear1', nn.Linear(in_features=120, out_features= 84)),
                ('relu4', nn.ReLU()),
                ('linear2', nn.Linear(in_features=84, out_features=10)),
            ])
        )
        self.flatten = nn.Flatten() 
    
    def forward(self, x):
        x = self.convs(x)
        x = self.fcn(self.flatten(x))
        return x