
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm 
import torch.nn.functional as F

import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance



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
            if (accuracy > best_accuracy and epoch > 8):
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


def train_dann(
    args, 
    classifier, 
    discriminator, 
    optimizer, 
    scheduler, 
    source_train_loader, 
    source_val_loader, 
    target_train_loader,
    target_val_loader,
    save_path,
):
    """
    Trains an Adaptation model assuming that the models are already on the approriate device
    """
    
    train_source_iter = ForeverDataIterator(source_train_loader)
    train_target_iter = ForeverDataIterator(target_train_loader)
    
    # print("Training the Domain Adversarial Network with backbone = ")
    domain_adv  = DomainAdversarialLoss(discriminator).to(args.device)
    
    best_accuracy = 0
    for epoch in range(args.adv_epochs):
        classifier.train()
        discriminator.train()
        avg_loss, avg_cls_loss, avg_transfer_loss = 0,0,0
        avg_domain_acc = 0
        wandb.log({f"Adversarial epoch": epoch})
        wandb.log({f"Adversarial lr": scheduler.get_last_lr()[0]})
        
        # need to check how many iterations are needed per epoch, approximately depends on the batchsize
        ITERATIONS_PER_EPOCH = max(len(source_train_loader), len(target_train_loader))
        for i in range(ITERATIONS_PER_EPOCH):
            x_s, labels_s = next(train_source_iter)[:2]
            x_t, = next(train_target_iter)[:1]
            
            if(x_s.shape[0] != x_t.shape[0]):
                optimizer.zero_grad()
                continue
            
            
            x_s = x_s.to(args.device)
            labels_s = labels_s.to(args.device)
            x_t = x_t.to(args.device)
            
      
            x = torch.cat((x_s, x_t), dim=0)
            y, f = classifier(x) # 128, 10 and 128, 1024
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)
  
            cls_loss = F.cross_entropy(y_s, labels_s)
            transfer_loss = domain_adv(f_s, f_t)
            domain_acc = domain_adv.domain_discriminator_accuracy
            loss = cls_loss + transfer_loss * args.trade_off
            
            avg_loss += loss.item()
            avg_cls_loss += cls_loss.item()
            avg_transfer_loss += transfer_loss.item()
            avg_domain_acc += domain_acc


            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
            pass
        
        
        if (epoch %1 == 0):
            avg_loss /= ITERATIONS_PER_EPOCH
            avg_cls_loss /= ITERATIONS_PER_EPOCH
            avg_transfer_loss /= ITERATIONS_PER_EPOCH
            avg_domain_acc /= ITERATIONS_PER_EPOCH
            
            wandb.log(
                {
                    "Full Training Loss": avg_loss,
                    "Source Classification Loss": avg_cls_loss,
                    "Transfer Loss": avg_transfer_loss,
                    "Domain Prediction Accuracy": avg_domain_acc,
                }
            )
            
            #calculate the validation accuracies of your DA model
            val_accuracy = validate(args, classifier, target_val_loader, model_name="DA Model")
            if(val_accuracy > best_accuracy):
                best_accuracy = val_accuracy
                torch.save(classifier.state_dict(), save_path)
                pass
            pass
            
            # Forward pass
    pass


def get_fisher_information(args, classifier, target_loader, num_classes, device):
    """
    Get the fisher information matrix for the model
    """
    
    classifier.train()
    fisher_matrix = {}
    count_per_class = [0 for i in range(num_classes)]
    gradients = []
    batch_fisher_info = []
    for i, (x, y) in enumerate(target_loader):
        x = x.to(device)
        y = y.to(device)
        y_hat,_ = classifier(x)
        log_probs = torch.nn.Softmax(dim=1)(y_hat)
        gradients = []
        # breakpoint()
        class_fisher_info_per_batch = []
        for j in range(num_classes):
            if (j in y):
                # find indices in x wher eth class is j 
                indices = torch.where(y == j)[0].tolist()
                # fisher information is the outer product of the gradient of log probabilites
                log_probs[indices, j].mean().backward() # should  I wait this using bayes rule?
                for param in classifier.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.flatten())
                    count_per_class[j] += 1
                    param.grad.data.zero_()
                    pass
            breakpoint()
            grad_log_prob = torch.cat(gradients) # this is too big
            class_fisher_info_per_batch.append(torch.outer(grad_log_prob, grad_log_prob))
            pass
        
        batch_fisher_info.append(torch.mean(torch.stack(class_fisher_info_per_batch), dim=0))
        
    fisher_matrix = torch.mean(torch.stack(batch_fisher_info), dim=0)
            
                # # append these 
                # sample_fisher = torch.autograd.grad(torch.log(y_hat[:, j]).mean(), classifier.parameters(), retain_graph=True)
                # # # calculate the gradient of the log of the probability of the correct class for whole batch 
                # # fisher_matrix[j] += torch.autograd.grad(torch.log(y_hat[:, j]).mean(), classifier.parameters(), retain_graph=True)
                # # # fisher_matrix[j] += torch.autograd.grad(y_hat[:, j].sum(), classifier.parameters(), retain_graph=True)
    return fisher_matrix


def compute_diagonal_fisher(args, classifier, target_loader, num_classes, device):
    """
    Get the diagonal of the fisher information matrix for the model
    """
    classifier.train()
    fisher_matrix = {}
    count_per_class = [0 for i in range(num_classes)]
    gradients = []
    batch_fisher_info = []
    for i, (x, y) in enumerate(target_loader):
        x = x.to(device)
        y = y.to(device)
        y_hat,_ = classifier(x) # not that y_hat by default is a non leaf with requires_grad = True
        y_hat = y_hat.requires_grad_()
        log_probs = torch.nn.functional.log_softmax(y_hat).requires_grad_()
        gradients = []

        class_fisher_info_per_batch = []
        """
        Calculate the fisher information of each type of example within each class
        """
        for j in range(num_classes):
            if (j in y):
                # find indices in x wher eth class is j 
                indices = torch.where(y == j)[0].tolist()
                # fisher information is the outer product of the gradient of log probabilites
                avg_log_prob = log_probs[indices, j].mean()
                avg_log_prob.backward(retain_graph=True) # should  I wait this using bayes rule?
                gradients = []
                for param in classifier.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.flatten())
                    count_per_class[j] += 1
                    pass
                classifier.zero_grad()
                grad_log_prob = torch.cat(gradients) # this is too big
                class_fisher_info_per_batch.append(torch.pow(grad_log_prob, 2))
            pass
        
        batch_fisher_info.append(torch.mean(torch.stack(class_fisher_info_per_batch), dim=0))
    fisher_matrix_diagonal = torch.mean(torch.stack(batch_fisher_info), dim=0)
    return fisher_matrix_diagonal 
    pass


def compute_per_example_fisher(args, classifier, target_loader, num_classes, device):
  for index, (x, y) in enumerate(target_loader):
    x = x.to(device)
    y = y.to(device)
    y_hat, _ = classifier(x)
    y_hat = torch.nn.functional.log_softmax(y_hat)
    y_hat = y_hat.requires_grad_()
    gradients = []
    for i in range(num_classes):
      if (i in y):
        indices = torch.where(y == i)[0].tolist()
        avg_log_prob = y_hat[indices, i].mean()
        avg_log_prob.backward(retain_graph=True)
        for param in classifier.parameters():
          if param.grad is not None:
            gradients.append(param.grad.flatten())
          pass
        pass
      pass
    pass
# find high confidence samples for classifier on the target loader

def get_high_confidence_samples(args, classifier, target_loader, device):
    classifier.eval()
    high_confidence_samples = []
    for i, (x, y) in enumerate(target_loader):
        x = x.to(device)
        y = y.to(device)
        y_hat, _ = classifier(x)
        y_hat = torch.softmax(y_hat, dim=1)
        for j in range(x.shape[0]):
            if (torch.max(y_hat[j]) > 0.9):
                high_confidence_samples.append((x[j], y[j]))
    return high_confidence_samples

# calculate empirical fisher information matrix only on these samples

def get_empirical_fisher_information(args, classifier, high_confidence_samples, device):
    classifier.eval()
    fisher_matrix = {}
    for i in range(10):
        fisher_matrix[i] = torch.zeros((1024, 1024)).to(device)
    for i, (x, y) in enumerate(high_confidence_samples):
        x = x.to(device)
        y = y.to(device)
        y_hat, _ = classifier(x)
        for j in range(10):
            if (j in y):
                fisher_matrix[j] += torch.autograd.grad(y_hat[:, j].sum(), classifier.parameters(), retain_graph=True)
    return fisher_matrix