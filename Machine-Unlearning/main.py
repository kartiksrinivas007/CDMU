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

from tllib.modules.domain_discriminator import DomainDiscriminator
from tllib.alignment.dann import DomainAdversarialLoss, ImageClassifier
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger
from tllib.utils.analysis import collect_feature, tsne, a_distance
if __name__ == "__main__":
    
    
    
    # initlaize the wandb system for logging and parse the arguments 

    args = parse_args()
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
    }
    )
    
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
            except:
                print("Could not load the source model, please pretrain the model")
                exit(0)
        #endregion

        
        # test the model on the test loader of the source dataset
        validate(args, source_model, ui.full_source_test_loader, "source_model", is_test=True)
    # get the transfer learning library inside the environment
    
    """
    Need to create these arguments to pass into the function fr domain adversarial trianing 
    args, classifier,discriminator, 
               train_loader, val_loader, 
               save_path
    """
    
    # obtain the sub-network or backbone of the source model
    new_source_backbone = get_network(
        "EncoderConvNet", channel=3, num_classes=10, im_size=(32, 32)
    )

    pool_layer = nn.Identity()
    classifier = ImageClassifier(new_source_backbone, num_classes,bottleneck_dim=1024,pool_layer=pool_layer).to(args.device) # might have to precompute this value
    disc = DomainDiscriminator(classifier.features_dim, hidden_size=1024).to(args.device)
    
    
    adv_optimizer = torch.optim.SGD(classifier.get_parameters() + disc.get_parameters(), lr = args.adv_lr, 
                                momentum=0.9, weight_decay=args.adv_weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.adv_lr * (1. + args.adv_lr_gamma * float(x)) ** (-args.adv_lr_decay))
    
    train_dann(args, classifier, disc, adv_optimizer, lr_scheduler, ui.full_source_train_loader, ui.full_source_val_loader, 
               ui.full_target_train_loader, ui.full_target_val_loader, os.path.join(CONFIG_SAVE_PATH, f"{args.source}_model.pth"))

    pass