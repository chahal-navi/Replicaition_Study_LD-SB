# IFM Dataset generators
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class IFM_dataset(Dataset):
  def __init__(self, gamma, dataset_size, num_coords):
    # by default the first coordinate is the one which will be generated as
    # the coordinate in which data is linearly separable
    self.gamma = gamma
    self.num_coords = num_coords
    self.size = dataset_size
    output = [1, -1]
    self.input_list = []
    self.output_list = []

    def non_linear_generator(y, coord_number):
      point_list = torch.arange(-(coord_number), (coord_number + 2), 2)
      # this is the list of the points where the coord_number_th featires would be centered
      # using a gausssain around them
      if y == 1:
        index_list = torch.arange(0, coord_number + 1 , 2)
        return (torch.normal(point_list[random.choice(index_list)], 0.1, (1,)).item())
      else:
        index_list = torch.arange(1, coord_number + 1, 2)
        return (torch.normal(point_list[random.choice(index_list)], 0.1, (1,)).item())


    # the dataset will be a binary output dataset with n input coords

    for i in range(self.size):
      y = random.choice(output)
      self.output_list.append(y)
      input_i = [self.gamma*y + torch.normal(0, 0.05, (1, )).item()]
      for j in torch.arange(2, self.num_coords + 1):
        input_i.append(non_linear_generator(y, j))
      self.input_list.append(input_i)


  def __len__(self):
    return (len(self.input_list))

  def __getitem__(self, idx):
    return (torch.tensor(self.input_list[idx], dtype = torch.float32), torch.tensor(self.output_list[idx], dtype = torch.float32))


class projection_generators(Dataset):
  def __init__(self, initial_generator, projection_vectors, dim):
    # The initial generator is an object of  the class IFM_dataset
    # The projection_vectors is a list of projection vectors obtained by the first k singular values
    # where K is the effective rank of the weight matrix
    self.initial_generator = initial_generator
    self.projection_vectors = [i/torch.linalg.norm(i) for i in projection_vectors]
    self.projection_vectors = torch.stack(self.projection_vectors, dim = 0)
    self.projection_matrix = torch.eye(dim) - torch.matmul(self.projection_vectors.T.cpu(), self.projection_vectors.cpu())
    self.input_list = []
    self.output_list = []
    for x, y in self.initial_generator:
      x = x.view(x.size(0), -1)
      self.input_list.append(torch.matmul(self.projection_matrix.cpu(), x.cpu()))
      self.output_list.append(y)

  def __len__(self):
    return(len(self.input_list))

  def __getitem__(self, idx):
    return (self.input_list[idx], self.output_list[idx])


class data_subsets(Dataset):
  def __init__(self, data, classes):
  # data is a dataset object whose subset is to be created and classes is a list of classes
  # in the subset
    super().__init__()
    self. data = data
    self.classes = classes
    self.input_list = []
    self.output_list = []
    for i in self.data:
      if i[1] in self.classes:
        self.input_list.append(i[0])
        self.output_list.append(i[1])

  def spurious_correlators(self, width):
    # width is the width of the patch of white/black pixels, we use horizontal
    # patches for spurious correllations::: white patch == class 0
    for i in torch.arange(len(self.input_list)):
      if self.output_list[i] == 0:
        self.input_list[i][0][0:width].fill_(1)
      else:
        self.input_list[i][0][0:width].fill_(0)


  def __len__(self):
    return (len(self.input_list))

  def __getitem__(self, idx):
    return(self.input_list[idx], self.output_list[idx])



