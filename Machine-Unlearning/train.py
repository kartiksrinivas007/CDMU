
import torch
import torch.nn as nn




# def pretrain_model(num_epochs, optimizer, model, train_loader, device, criterion):
#     model.train()
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device)
            
#             # Forward pass