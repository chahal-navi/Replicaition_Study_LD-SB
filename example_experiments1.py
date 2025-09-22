
# this example shows training models using the initialisation schemes on the ifm dataset and then
# comparing the rank evolution duynamics of various models. 


# model 1 training starts
epoch = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
rich_model_num1 = rich_model(20, 400)
optimizer = torch.optim.SGD(rich_model_num1.parameters(), lr = 10e-3, momentum = 0.9)
training_dataset = IFM_dataset(gamma = 2,dataset_size = 10000, num_coords = 20)
training_data_loader = DataLoader(training_dataset, batch_size = 64, shuffle = True)
rich_model_num1.to(device)


def loss_func(output, target):
    return torch.mean(torch.nn.functional.softplus(-target*output))
weight_matrix_rich_model_num1 = []
while epoch < 100:
  print(f"Epoch {epoch + 1}\n-------------------------------")
  weight_matrix_rich_model_num1.append(rich_model_num1.layer1.weight.data.cpu().clone())
  
  train_loop(training_data_loader, rich_model_num1, loss_func, optimizer, device)
  epoch += 1
print("done")


#model 2 training starts
epoch = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
rich_model_num12 = rich_model(20, 600)
optimizer = torch.optim.SGD(rich_model_num12.parameters(), lr = 10e-3, momentum = 0.9)
training_dataset = IFM_dataset(gamma = 2,dataset_size = 10000, num_coords = 20)
training_data_loader = DataLoader(training_dataset, batch_size = 64, shuffle = True)
rich_model_num12.to(device)


def loss_func(output, target):
    return torch.mean(torch.nn.functional.softplus(-target*output))
weight_matrix_rich_model_num12 = []
while epoch < 100:
  print(f"Epoch {epoch + 1}\n-------------------------------")
  weight_matrix_rich_model_num12.append(rich_model_num12.layer1.weight.data.cpu().clone())
  
  train_loop(training_data_loader, rich_model_num12, loss_func, optimizer, device)
  epoch += 1
print("done")

#model 3 training starts

epoch = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
rich_model_num13 = rich_model(20, 100)
optimizer = torch.optim.SGD(rich_model_num13.parameters(), lr = 10e-3, momentum = 0.9)
training_dataset = IFM_dataset(gamma = 2,dataset_size = 10000, num_coords = 20)
training_data_loader = DataLoader(training_dataset, batch_size = 64, shuffle = True)
rich_model_num13.to(device)


def loss_func(output, target):
    return torch.mean(torch.nn.functional.softplus(-target*output))
weight_matrix_rich_model_num13 = []
while epoch < 100:
  print(f"Epoch {epoch + 1}\n-------------------------------")
  weight_matrix_rich_model_num13.append(rich_model_num13.layer1.weight.data.cpu().clone())
  
  train_loop(training_data_loader, rich_model_num13, loss_func, optimizer, device)
  epoch += 1
print("done")

# analysing rank using the Matrix_analyser class defined in data_visualizers.py

plott = Matrix_analyser([weight_matrix_rich_model_num13, weight_matrix_rich_model_num1, weight_matrix_rich_model_num12])
plott.effective_rank()

