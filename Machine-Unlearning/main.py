"""
Machine Unlearning via source free domain adaptation is going to be experimented with
Baselines : Golathkar 
"""

# import argparse
from utility_classes import *
from utils import *
from parser_utils import *
from Dataset_utils import *
from parser_utils import *
import pickle
import wandb
from train import *
# from salad import solver
import sys
from torch.optim.lr_scheduler import LambdaLR

import gc
from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance

from torch.utils.data import DataLoader, Subset
if __name__ == "__main__":
    
    
    
    # initlaize the wandb system for logging and parse the arguments 

    args = parse_args()
    if (args.wandb):
        
        wandb.init(

        project="CDMU",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.source_lr,
        "architecture": "ConvNet",
        "dataset": "Digits",
        "epochs": args.source_epochs,
        "source" : args.source,
        "target" : args.target,
        "adv_lr" : args.adv_lr,
        "adv_epochs" : args.adv_epochs,
        "adv_weight_decay" : args.adv_weight_decay,
        "algorithm" : args.algorithm,
        "lambda_fisher" : args.lambda_fisher,
        "num_forget" : args.num_forget,
        "lambda_ewc" : args.lambda_ewc,
        "device" : args.device,
        "seed" : args.seed,
        "ft_lr" : args.ft_lr,
        "fine_tune_epochs" : args.fine_tune_epochs,
        "lambda_pseudo" : args.lambda_pseudo,
        "lambda_rl": args.lambda_rl,
        }
        )

    
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
    if (CUDA_VISIBLE_DEVICES is not None):
        device_ids = [int(i) - 1 for i in CUDA_VISIBLE_DEVICES.split(",")]
        args.device = torch.device('cuda:0') # use the first homing centre as the device
    else:
        args.device = 'cpu'
        
    print('using cuda devices = ', args.device)
    print("Device_ids = ", device_ids)
    
    # print("Device name =",  torch.cuda.get_device_name(args.device))
    
    if (not args.resume):
        print("Creating a new instance")
        information = return_domain_information(args)
        ui = UnlearningInstance(args, information)
        # dump the ui instance into a checkpoint folder using parquet
        try: 
            checkpoint_object(ui, args)
        except:
            print("Could not save the checkpoint")
            exit(0)
        
    else:
        try:
            ui =load_checkpoint_object(args, "Unlearning_Instance")
            print("Loaded the checkpoint for Unlearning_Instance")
        except:
            print("Could not find Ui instance to resume, loading new instance")
            information = return_domain_information(args)
            ui = UnlearningInstance(args, information)
            checkpoint_object(ui, args)
        pass
    
    CONFIG_SAVE_PATH = os.path.join(args.save_path, f"Joint/{args.source}")
    SOURCE_PRETRAIN_SAVE_PATH = os.path.join(args.save_path, f"Pretrain")
    
    if (not os.path.exists(CONFIG_SAVE_PATH)):
        os.makedirs(CONFIG_SAVE_PATH)
        
    if (not os.path.exists(SOURCE_PRETRAIN_SAVE_PATH)):
        os.makedirs(SOURCE_PRETRAIN_SAVE_PATH)
    
    
    if (args.verbose):
        ui.display_variables()
    # now we bring a model in that can handle this information
    
    if (args.dset == "digits"):
        num_classes = 10
        
        """
        Get the source model and train it on the source dataset
        """
        #region
        source_model = get_network("ConvNet", channel = 3, num_classes = 10, im_size=(32, 32))
        source_model.to(args.device)
        optimizer = torch.optim.Adam(source_model.parameters(), lr = args.source_lr)
        if (args.pretrain):
            model, epoch = train_model(args, "source_model", args.source_epochs, optimizer, source_model,
                        ui.full_source_train_loader, ui.full_source_val_loader, 
                        os.path.join(SOURCE_PRETRAIN_SAVE_PATH, f"{args.source}_model.pth"))
            
            print("Best Epoch for the source model is ", epoch)
        else:
            try:   
                source_model.load_state_dict(torch.load(os.path.join(SOURCE_PRETRAIN_SAVE_PATH, f"{args.source}_model.pth")))
                print("Loaded the source model")
                source_model = source_model.to(args.device)
            except:
                print("Could not load the source model, please pretrain the model")
                exit(0)
                
            
        new_source_backbone = get_network(
        "EncoderConvNet", channel=3, num_classes=10, im_size=(32, 32),
        )
        pool_layer = nn.Identity()
        
        #endregion
    elif args.dset == "office":
        num_classes = 65
        
        """
        Get the source model and train it on the source dataset
        """
        new_source_backbone = get_network(
        "_resnet_50_backbone", channel=3, num_classes=65, im_size=(224, 224),
        )
        pool_layer = nn.Identity()
        classifier = ImageClassifier(new_source_backbone, num_classes,bottleneck_dim=1024,pool_layer=pool_layer).to(args.device)
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad,  classifier.parameters()), lr = args.source_lr)
        if (args.pretrain):
            model, epoch = train_model(args, args.source_epochs, optimizer, classifier,
                        ui.full_source_train_loader, ui.full_source_val_loader, 
                        os.path.join(SOURCE_PRETRAIN_SAVE_PATH, f"{args.source}_model.pth"), True) 
            print("Best Epoch for the source model is ", epoch)
        else:
            try:   
                classifier = torch.load(os.path.join(SOURCE_PRETRAIN_SAVE_PATH, f"{args.source}_model.pth"))
                print("Loaded the source model")
                classifier = classifier.to(args.device)
            except:
                print("Could not load the source model, please pretrain the model")
                exit(0) 
        validate(args, classifier , ui.target_forget.remember_test_loader, "Pre-trained model acc on the target remember") 
        pass
   
    torch.cuda.empty_cache()
    gc.collect()
    
    if (args.adv_train):
        
        # obtain the sub-network or backbone of the source model
        
        # backbone -> bottleneck -> head is the total flow
        # backbone needs an out-features property that will be accessed for making the bottleneck!
        
        validate(args, classifier, ui.target_forget.remember_test_loader, "Pre-adapted model acc on the Target Remember")
        disc = DomainDiscriminator(classifier.features_dim, hidden_size=1024).to(args.device)
        
        
        adv_optimizer = torch.optim.SGD(classifier.get_parameters() + disc.get_parameters(), lr = args.adv_lr, 
                                    momentum=0.9, weight_decay=args.adv_weight_decay, nesterov=True)
        lr_scheduler = LambdaLR(optimizer, lambda x: args.adv_lr * (1. + args.adv_lr_gamma * float(x)) ** (-args.adv_lr_decay))
        
#        breakpoint()
        """
        The model needs to learn on the full source data and adapt only to the target remember loader.
        After which the model will be tested on the following four datasets 
        1. The source remember classes
        2. The source forget classes
        3. The target remember classes
        4. The target forget classes (the ones not trained upon in the train_dann)

        Train DANN function only tests on type (3) during the DANN adaptation and the (1,2) joint source
        """

        train_dann(args, classifier, disc, adv_optimizer, lr_scheduler, ui.full_source_train_loader, ui.full_source_val_loader, 
                ui.target_forget.remember_train_loader, ui.target_forget.remember_test_loader, os.path.join(CONFIG_SAVE_PATH, f"{args.target}_model.pth"))
        
    else:
        try:
            classifier = (torch.load(os.path.join(CONFIG_SAVE_PATH, f"{args.target}_model.pth")))
            print("Loaded the Domain Adversarially trained classifier")
        except:
            print("Could not load the classifier, please DA train the model")
            exit(0)
        pass
    """    
    if (args.algorithm == "ewc"):
        regularized_fine_tune(args, classifier, classifier_copy, new_optim, small_forget_loader, num_classes,ui.full_target_train_loader, small_forget_loader,args.fisher, fisher=fisher_diagonal)
    elif (args.algorithm == "newton"):
        fine_tune(args, classifier, adam_optim, small_forget_loader, num_classes, args.fisher, fisher=fisher_diagonal)
    elif (args.algorithm == "pseudo"):
        examples, labels = get_high_confidence_samples(args, classifier, ui.full_target_train_loader, args.device)
        pseudo_target_loader = DataLoader(TensorDataset(examples, labels), batch_size=32, shuffle=True)
        pseudo_optimzer = torch.optim.SGD(classifier.get_parameters(), lr = args.ft_lr)
        pseudo_ascent(args, classifier, classifier_copy, pseudo_optimzer, pseudo_target_loader, small_forget_loader, num_classes)

    else:
        examples, labels = get_high_confidence_samples(args, classifier, ui.full_target_train_loader, args.device)
        pseudo_target_loader = DataLoader(TensorDataset(examples, labels), batch_size=32, shuffle=True)
        # breakpoint()
        
        num_params = sum([param.numel() for param in classifier.parameters()])
        noise_covariance = torch.rand(num_params, requires_grad=True, device=args.device)
        pseudo_optimzer = torch.optim.SGD([noise_covariance], lr = args.ft_lr)
        validate(args, classifier, pseudo_target_loader, "Before using unlearning model on Target", is_test=True)
        classifier, fixed_model, noise_covariance = reparametrized_lagrangian(args, classifier, classifier_copy, pseudo_optimzer, pseudo_target_loader, small_forget_loader, noise_covariance)
        
        indices = list(range(num_params))
        for param in fixed_model.parameters():
            param.data = param.data + noise_covariance[indices[:param.numel()]].reshape(param.shape)
            indices = indices[param.numel():]
        
        
        new_model_norm = torch.norm(torch.cat([param.flatten() for param in classifier.parameters()]))
        fixed_model_norm = torch.norm(torch.cat([param.flatten() for param in fixed_model.parameters()]))
        validate(args, fixed_model, ui.full_target_test_loader, "Noisy model on Target", is_test=True)
        validate(args, fixed_model, small_forget_loader, "Noisy model on Forgetset", is_test=True)
        print("Model Norm before and after update", model_norm, new_model_norm)
                
        
    new_model_norm = torch.norm(torch.cat([param.flatten() for param in classifier.parameters()]))
    print("Model Norm before and after update", model_norm, new_model_norm)
    # breakpoint()
    # now validate the model on the target domain and the forgetset
""" 
