import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 1, padding = 2,),# stride = 1, padding = (kernel_size-1)/2 = (5-1)/2
        nn.ReLU(), #()
        nn.MaxPool2d(kernel_size = 2),
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(64, 128, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.out = nn.Linear(128*4*4, 42)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = x.view(x.size(0), -1)
    output = self.out(x)
    return F.log_softmax(output), x
