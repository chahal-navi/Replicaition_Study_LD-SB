# Data Visualiser
import matplotlib.pyplot as plt


class DataVisualizer:
  def __init__(self, dataset):
    # the dataset variable is an object inheriting from IFM_dataset/ torch.utils.Dataset
    self.dataset = dataset
    self.num_plots = self.dataset.num_coords

  def plotter(self, coord_list):
    # coord_number will be a list of the coords to be plotted, by default all will be done
    input_list = torch.tensor(self.dataset.input_list)
    for i in coord_list:
      x_data = input_list[:, i - 1]
      y_data = self.dataset.output_list
      plt.scatter(x_data, y_data)
      plt.show()
# The following class will be used to ananlyse the evolution dynamics of the weight matrices
class Matrix_analyser():
  def __init__(self, matrix_list):
    # Matrix list is the list of matrices for all the epochs inside the training
    # loop
    self.num_epoch = len(matrix_list[0])
    self.matrix_list = matrix_list
    self.effective_rank_list = []

  def effective_rank(self):
    for model_num, matrix in enumerate(self.matrix_list):
      eff = []
      print(model_num)
      for idx, i in enumerate(matrix):
        u, s, v = torch.svd(torch.tensor(i))
        s = s*s
        summ = torch.sum(s)
        s = s/summ
        eff.append(torch.exp(-torch.sum(s*torch.log(s))))
        if idx == 99:
          self.effective_rank_list.append(torch.exp(-torch.sum(s*torch.log(s))))
        
      plt.plot(torch.arange(self.num_epoch), eff, label = f"Hidden Units:{(model_num+1)*200}")
    plt.title("Rank Evolution of Lazy Initialisation in Training")
    plt.legend()
    plt.show()

