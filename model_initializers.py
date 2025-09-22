# model with various initialisations
from torch import nn
import numpy as np

class rich_model(nn.Module):
  def __init__(self, num_coords, m):
        super().__init__()
        self.num_coords = num_coords
        self.m = m
        self.layer1 = nn.Linear(num_coords, m)
        with torch.no_grad():
            for i in range(m):
                vector = torch.randn(num_coords + 1)
                vector = vector / torch.norm(vector)
                self.layer1.weight.data[i] = vector[:num_coords]
                self.layer1.bias.data[i] = vector[num_coords]

        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(m, 1, bias=False)

        with torch.no_grad():
            # Scale by 1/m in initialization, not forward pass
            signs = torch.tensor(np.random.choice([1, -1], size=m), dtype=torch.float32)
            self.layer3.weight.data = signs
            self.layer3.weight.data = signs.view(1, -1)
            



  def forward(self, x):
    if isinstance(x, list):
      x = torch.stack(x)
    return(self.layer3(self.layer2(self.layer1(x)))/self.m)



class lazy_model(nn.Module):
  def __init__(self, num_coords, m):
    super().__init__()
    self.num_coords = num_coords
    # Here m is the number of hidden units in layer 2
    self.m = m
    self.layer1 = nn.Linear(num_coords, self.m)
    with torch.no_grad():
      
      self.layer1.weight.data = torch.normal(0,1/self.num_coords , size = self.layer1.weight.data.shape)
      self.layer1.bias.data = torch.zeros_like(self.layer1.bias.data)
    self.layer2 = nn.ReLU()
    self.layer3 = nn.Linear(self.m, 1, bias = False)

    with torch.no_grad():
      self.layer3.weight.data = torch.normal(0, 1/self.m, size = self.layer3.weight.data.shape)

  def forward(self, x):

    return(self.layer3(self.layer2(self.layer1(x))))


# Model for images
class rich_model_flattened(nn.Module):
  def __init__(self, num_coords, m):
        super().__init__()
        self.num_coords = num_coords
        self.m = m
        self.layer1 = nn.Linear(num_coords, m)
        with torch.no_grad():
            for i in range(m):
                vector = torch.randn(num_coords + 1)
                vector = vector / torch.norm(vector)
                self.layer1.weight.data[i] = vector[:num_coords]
                self.layer1.bias.data[i] = vector[num_coords]

        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(m, 1, bias=False)

        with torch.no_grad():
            
            signs = torch.tensor(np.random.choice([1, -1], size=m), dtype=torch.float32)
        self.layer3.weight.data = signs.view(1, -1)
        self.layer4 = nn.Sigmoid()

  def forward(self, x):
    x = x.view(x.size(0), -1)
    return (self.layer4(self.layer3(self.layer2(self.layer1(x)))/self.m))


class lazy_model_flattened(nn.Module):
  def __init__(self, num_coords, m):
    super().__init__()
    self.num_coords = num_coords
    # Here m is the number of hidden units in layer 2
    self.m = m
    self.layer1 = nn.Linear(num_coords, self.m)
    with torch.no_grad():
      
      self.layer1.weight.data = torch.normal(0,1/self.num_coords , size = self.layer1.weight.data.shape)
      self.layer1.bias.data = torch.zeros_like(self.layer1.bias.data)
    self.layer2 = nn.ReLU()
    self.layer3 = nn.Linear(self.m, 1, bias = False)

    with torch.no_grad():
      self.layer3.weight.data = torch.normal(0, 1/self.m, size = self.layer3.weight.data.shape)
    self.layer4 = nn.Sigmoid()
  def forward(self, x):
    x = x.view(x.size(0), -1)
    return(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
