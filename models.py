import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DQN(nn.Module):
  def __init__(self, embedding_size):
    super(DQN, self).__init__()
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.head = nn.Linear(448, embedding_size)

    self.initialize_weights()

  def initialize_weights(self):
    conv_layers = [v for k,v in self._modules.iteritems() if 'conv' in k]
    for layer in conv_layers:
      init.xavier_uniform(layer.weight)
    init.xavier_uniform(self.head.weight)

  def forward(self, data):
    output = F.selu(self.conv1(data))
    output = F.selu(self.conv2(output))
    output = F.selu(self.conv3(output))
    output = self.head(output.view(output.size(0), -1))
    return output
