
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm 
import torch.nn.functional as F
import copy
import utils
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

import gc

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
            if(args.wandb):
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
    if (args.wandb):
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
            # breakpoint()
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
    classifier.eval()
    r = torch.cuda.memory_reserved(0)
    print(f"Reserved memory: {r}")
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    a = torch.cuda.memory_allocated(0)
    print(f"Allocated memory: {a}")

    fisher_matrix = {}
    count_per_class = [0 for i in range(num_classes)]
    gradients = []
    batch_fisher_info = torch.zeros_like(torch.cat([param.flatten() for param in classifier.parameters()]))
    for i, (x, y) in enumerate(target_loader):
        x = x.to(device)
        y = y.to(device)
        y_hat = classifier(x) # not that y_hat by default is a non leaf with requires_grad = True
        y_hat = y_hat.requires_grad_()
        log_probs = torch.nn.functional.log_softmax(y_hat, dim=1).requires_grad_()
        gradients = []
        wandb.log({"Memory allocated": torch.cuda.memory_allocated(0)})
        torch.cuda.empty_cache()
        class_fisher_info_per_batch = torch.zeros_like(torch.cat([param.flatten() for param in classifier.parameters()]))
        """
        Calculate the fisher information of each type of example within each class
        """
        for j in range(num_classes):
            if (j in y):
                # find indices in x wher eth class is j 
                # breakpoint()
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
                # grad_log_prob =  # this is too big
                # class_fisher_info_per_batch.append(torch.pow(grad_log_prob, 2))
                class_fisher_info_per_batch += torch.abs(torch.cat(gradients))
                torch.cuda.empty_cache()
            pass
        
        batch_fisher_info += class_fisher_info_per_batch/target_loader.batch_size
    fisher_matrix_diagonal =  batch_fisher_info/len(target_loader)
    return fisher_matrix_diagonal 
    pass



    # define a new fine tune function that fine_tunes the model on the forget data
def update_strategy(param, fisher_update, args):
    # breakpoint()
    threshold = 1e-2
    update = torch.zeros_like(param.grad)
    positions  = torch.where(fisher_update <= threshold)
    update[positions] = 0
    sensitive_positions = torch.where(fisher_update > threshold)
    #rescale the sensitive positions ot become a factor between 0 and 1 fraction of the param_grad
    inv_fisher = 1/fisher_update[sensitive_positions]
    # if (torch.isnan(inv_fisher).any()):
    #     # breakpoint()
    # if (torch.isinf(inv_fisher).any()):
    #     # breakpoint()
    inv_fisher = (args.lambda_fisher * inv_fisher).clip(0, 10)
    update[sensitive_positions] = (- 1) * param.grad[sensitive_positions]
    wandb.log({"Fraction of Sensitive updates": sensitive_positions[0].shape[0]/param.numel()})
    return update
    pass
   
   
def fine_tune(args, classifier, optimizer, forget_loader, num_classes, is_fisher=False, fisher=None):
    classifier.train()
    device = args.device
    criterion = nn.CrossEntropyLoss()
    # fisher_copy = fisher.copy()
    augmented_fisher = copy.deepcopy(fisher)
    for epoch in range(args.fine_tune_epochs):
        for index, (x, y) in enumerate(forget_loader):
            x = x.to(device)
            y = y.to(device)
            y_hat, _ = classifier(x)
            loss = -1 * criterion(y_hat, y) # take an ascent on the forget samples
            optimizer.zero_grad()
            loss.backward()
            if not is_fisher:
                optimizer.step()
            else:
                del augmented_fisher
                torch.cuda.empty_cache()
                gc.collect()
                augmented_fisher = copy.deepcopy(fisher)
                update_norms = 0
                max_param_update_norm = 0
                max_param_grad = 0
                for i, param in enumerate(classifier.parameters()):
                    # reshape the fisher to the size of param afer taking the numel
                    # give a certain minimum value to each element of the fisher
                    fisher_update = augmented_fisher[:param.numel()].reshape(param.shape)
                    update = update_strategy(param, fisher_update, args)
                    
                    max_param_update_norm = max(max_param_update_norm, torch.norm(update))
                    max_param_grad = max(max_param_grad, torch.norm(param.grad))
                    
                    param.grad = param.grad  + update
                    update_norms += torch.norm(update)
                    augmented_fisher = augmented_fisher[param.numel():]
                    pass
                optimizer.step()
                pass
                # if(update_norms > 1e6):
                #     breakpoint()
                wandb.log({"Update Norms": update_norms})
                wandb.log({"Max Update Norm": max_param_update_norm})
                wandb.log({"Max Grad Norm": max_param_grad})
            pass
        pass
    return classifier
    pass

def regularized_fine_tune(args, classifier, fixed_model, optimizer, forget_loader, num_classes, pseudo_target_loader, small_forget_loader,is_fisher=False, fisher=None):
    classifier.train()
    device = args.device
    criterion = nn.CrossEntropyLoss()
    augmented_fisher = copy.deepcopy(fisher)
    for epoch in range(args.fine_tune_epochs):
        for index, (x, y) in enumerate(forget_loader):
            x = x.to(device)
            y = y.to(device)
            y_hat, _ = classifier(x)
            loss = -1 * criterion(y_hat, y)
            optimizer.zero_grad()
            # breakpoint()
            # add a norm difference between the parameters of the fixed_model
            # and the classifier
            for param, fixed_param in zip(classifier.parameters(), fixed_model.parameters()):
                loss += args.lambda_ewc * torch.norm(param - fixed_param)
                
                      
            if (index % (len(forget_loader) // 2 ) == 0):
                validate(args, classifier, pseudo_target_loader, model_name="Pseudo model on Target")
                validate(args, classifier, small_forget_loader, model_name="Pseudo model on Forget")
                classifier.train()
            
            # breakpoint()
            loss.backward()
            optimizer.step()
    pass

    # define a function to compute the fisher information matrix per example

def get_high_confidence_samples(args, classifier, target_loader, device):
    classifier.eval()
    high_confidence_samples = []
    # high_confidence_indices = []
    
    for i, (x, y) in enumerate(target_loader):
        x = x.to(device)
        y = y.to(device)
        y_hat = classifier(x)
        y_hat = torch.softmax(y_hat, dim=1)
        y_estimated = torch.argmax(y_hat, dim=1)
        for j in range(x.shape[0]):
            if (torch.max(y_hat[j]) > 0.9):
                high_confidence_samples.append((x[j], y_estimated[j]))
    
    labels_concat = torch.stack([sample[1] for sample in high_confidence_samples])
    examples_concat = torch.stack([sample[0] for sample in high_confidence_samples])
    print("Number of high confidence samples: ", len(high_confidence_samples))
    print("Fraction of high confidence samples: ", len(high_confidence_samples)/len(target_loader.dataset))
    return examples_concat, labels_concat


def pseudo_ascent(args, classifier, fixed_model ,optimizer,  pseudo_target_loader, small_forget_loader, num_classes, is_fisher=False):
    classifier.train()
    device = args.device
    criterion = nn.CrossEntropyLoss()
    
    psuedo_iter = ForeverDataIterator(pseudo_target_loader)
    forget_iter = ForeverDataIterator(small_forget_loader)
    
    for epoch in range(args.fine_tune_epochs):
        ITERATIONS_PER_EPOCH = max(len(pseudo_target_loader), len(small_forget_loader))
        for index in range(ITERATIONS_PER_EPOCH):
            x_p, y_p = next(psuedo_iter)[:2]
            x_f, y_f = next(forget_iter)[:2]
            
            x_p = x_p.to(device)
            y_p = y_p.to(device)
            x_f = x_f.to(device)
            y_f = y_f.to(device)
            
            y_p_hat, _ = classifier(x_p)
            y_f_hat, _ = classifier(x_f)
            # breakpoint()
            loss_f = - args.lambda_pseudo * criterion(y_f_hat, y_f)
            loss_p = criterion(y_p_hat, y_p) 
            loss_reg = torch.tensor(0.0).to(device)
            for param, fixed_param in zip(classifier.parameters(), fixed_model.parameters()):
                loss_reg += args.lambda_ewc * torch.norm(param - fixed_param)
                
            wandb.log({"Pseudo Loss": loss_p.item()})
            wandb.log({"Forget Loss": loss_f.item()})
            wandb.log({"Reg Loss": loss_reg.item()})
            wandb.log({"Total Loss": loss_p.item() + loss_f.item() + loss_reg.item()})
            # wandb.log({"Loss_ratio forget": loss_f.item()/loss_p.item()})
            # wandb.log({"Loss_ratio reg": loss_reg.item()/loss_p.item()})
            
            if (index %200 == 0):
                validate(args, classifier, pseudo_target_loader, model_name="Pseudo model on Target")
                validate(args, classifier, small_forget_loader, model_name="Pseudo model on Forget")
                classifier.train()
            
            loss = loss_f + loss_p + loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass
        pass
    return classifier
    
    pass

def baseline_ascent(args, classifier, small_forget_loader, optimizer, num_classes):
    classifier.train()
    device = args.device
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.fine_tune_epochs):
        ITERATIONS_PER_EPOCH = len(small_forget_loader)
        for index in range(ITERATIONS_PER_EPOCH):
            x_f, y_f = next(small_forget_loader)[:2]
            x_f = x_f.to(device)
            y_f = y_f.to(device)
            y_f_hat, _ = classifier(x_f)
            loss_f = criterion(y_f_hat, y_f)
            optimizer.zero_grad()
            loss_f.backward()
            optimizer.step()
            pass
        pass
    return classifier
    pass


# calculate empirical fisher information matrix only on these samples
def reparametrized_lagrangian(args, classifier, fixed_model, optimizer, pseudo_target_loader, small_forget_loader, noise_covariance):
    """
    Noise covariance is assumed to be on the same device as the classifier
    """
    classifier.train()
    device = args.device
    criterion = nn.CrossEntropyLoss()
    num_params = sum([param.numel() for param in classifier.parameters()])
    psuedo_iter = ForeverDataIterator(pseudo_target_loader)
    # noise_covariance = noise_covariance *1e-5
    # breakpoint()
    for epoch in range(args.fine_tune_epochs):
        ITERATIONS_PER_EPOCH = len(pseudo_target_loader)
        #reset the classifer parameters
        # classifier.load_state_dict(fixed_model.state_dict()) # try this if it works
        for index in tqdm(range(ITERATIONS_PER_EPOCH), desc = "Training the RL"):
            x_p, y_p = next(psuedo_iter)[:2]
            
            x_p = x_p.to(device)
            y_p = y_p.to(device)

            
            # sample a random gaussian of size num_params between 0 and 1
            # breakpoint()
            loss_product = torch.tensor(0.0, requires_grad=True, device=device)
            entry_list = list(range(num_params))
            
            update_vector = 1e-3*torch.randn(num_params, device=device) * noise_covariance
            for param in classifier.parameters():
                param.data = param.data + (update_vector[entry_list[:param.numel()]]).reshape(param.shape)
                entry_list = entry_list[param.numel():]
            
            if(index %20 == 0):
                validate(args, classifier, pseudo_target_loader, model_name="RL Model on Pseudo Samples")
                validate(args, classifier, small_forget_loader, model_name="RL model on Forget")
            loss_product = torch.norm(noise_covariance, 1)
            classifier.train()
            y_p_hat, _ = classifier(x_p)
            
            
 
            # breakpoint()
            # loss_f = - args.lambda_pseudo * criterion(y_f_hat, y_f)
            loss_p = criterion(y_p_hat, y_p) 
            loss_det = -1 * args.lambda_rl * loss_product
            # for param, fixed_param in zip(classifier.parameters(), fixed_model.parameters()):
            #     loss_reg += args.lambda_ewc * torch.norm(param - fixed_param)
                
            wandb.log({"Pseudo Loss": loss_p.item()})
            wandb.log({"Det Loss": loss_det.item()})
            wandb.log({"Total Loss": loss_p.item() + loss_det.item()})
            wandb.log({"Loss_ratio RL": loss_det.item()/loss_p.item()})
            wandb.log({"Noise 2 norm ": torch.norm(noise_covariance, 2)})

            
            loss = loss_p + loss_det
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for p in optimizer.param_groups[0]['params']:
                # print(p.grad)
                wandb.log({"Gradient Noise Norm": torch.norm(p.grad)})
                       #resset the parameters
            entry_list = list(range(num_params))
            for param in classifier.parameters():
                param.data = param.data - (update_vector[entry_list[:param.numel()]]).reshape(param.shape)
                entry_list = entry_list[param.numel():]
            
            torch.cuda.empty_cache()
            gc.collect()
            pass
        pass
    return classifier, fixed_model, noise_covariance

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