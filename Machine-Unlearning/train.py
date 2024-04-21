
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm 
#setup the training of the model with wandb and resumption of the training if needed

# def pretrain_model(num_epochs, optimizer, model, train_loader, device, criterion):
#     model.train()
#     for epoch in range(num_epochs):
#         for i, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device)
            
#             # Forward pass



def train_model(args, model_name, num_epochs, optimizer, model, train_loader, val_loader, save_path):
    """
    Trains the model assuming that the model is already on the correct device 
    with the `model_name` used for the loggin progress on wandb
    """
    device = args.device
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        for index, (images, labels) in tqdm(enumerate(train_loader), desc = "Training the Source Model"):
            images = images.to(device)
            labels = labels.to(device)
            # breakpoint()
            logits = model(images) # 64, 10
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if (epoch % 1 == 0):
            print(f"Epoch: {epoch} Iteration: {index} Loss: {loss.item()}")
            wandb.log({f"{model_name}_training_loss": loss.item()})
            wandb.log({f"{model_name}_epoch": epoch})
            accuracy = validate(args, model, val_loader, model_name)
            if (accuracy > best_accuracy and epoch > 5):
                best_accuracy = accuracy
                torch.save(model.state_dict(), save_path)
                best_epoch = epoch  
            pass
        
    return model, best_epoch
                            
def validate(args, model, val_loader, model_name, is_test=False):
    """
    Validates the model on the validation dataset assuming that the model is already on the specified device
    """
    device = args.device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, 1) # need to see the shape to verify this
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pass
    accuracy = 100 * correct / total

    log_title = "Validation" if not is_test else "Test"
    print(f"Accuracy of {model_name} on the  {log_title} dataset is {accuracy}")
    wandb.log({f"{model_name}_{log_title}_accuracy": accuracy})
    return accuracy




def pretrain_baselines(args, model_name, num_epochs, model, train_loaders, val_loaders):
    
    pass