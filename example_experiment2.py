# this files shows training the models on MNIST dataset by first modifying it to only contain
# the first two classes and then adding a patch of pixels ( width 1 ) using the spurious_correlators function in the
# data_subsets class defined in the data_generators.py file

import torchvision
from torchvision import transforms
from google.colab.patches import cv2_imshow
transform = transforms.ToTensor()

data = torchvision.datasets.MNIST(root = '/content/mnistt', transform = transform, train = True, download = True)

b_img = data_subsets(data, [0, 1])

b_img.spurious_correlators(1)  # the arguement defines the vertical range of the patch, thus 1 here means a horizontal patch 
# which is 1x28 pixels in dimension in a 28x28 pixels image.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
rich_model_num3 = rich_model_flattened(784, 200)
optimizer = torch.optim.SGD(rich_model_num3.parameters(), lr = 10e-4, momentum = 0.9)
training_dataloader = DataLoader(b_img, batch_size = 64, shuffle = True)
loss_func = nn.BCELoss()
epoch = 0
rich_model_num3.to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=500,
    eta_min=1e-6
)
weight_matrix_rich_model_num3 = []
while epoch < 100:
  print(f"Epoch {epoch + 1}\n-------------------------------")
  weight_matrix_rich_model_num3.append(rich_model_num3.layer1.weight.data.cpu().clone())
  train_loop(training_dataloader, rich_model_num3, loss_func, optimizer, device)
  
  epoch += 1
print("done")

# we can train more models and analyse their differences as described in example_experiments1.py file 

