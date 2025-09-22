# Training the models


def train_loop(dataloader, model, loss_func, optimizer, device):
    size = len(dataloader.dataset)

    model.train()
    for batch, (x, y) in enumerate(dataloader):
        if isinstance(x, list):
            x = torch.stack(x)
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        y = y.unsqueeze(1)
        x = x.to(device)
        y = y.to(device)
        loss = loss_func(model(x), y)
        loss.backward()
        total_grad_norm = 0
        #for name, param in model.named_parameters():
          #if param.grad is not None:
            #grad_norm = param.grad.norm().item()
            #total_grad_norm += grad_norm
            #if batch % 100 == 0:

                #print(f"{name} gradient norm: {grad_norm:.6f}")
    
        if batch % 100 == 0:
          print(f"models output {model(x[0])}")
          #print(f"Total gradient norm: {total_grad_norm:.6f}")
    
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * 64 + len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
