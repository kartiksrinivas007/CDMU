import argparse
import os
import torch
import random
import numpy as np

def parse_args():
    
    parser = argparse.ArgumentParser(description='Machine Unlearning')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    parser.add_argument('--dset', type=str, default='digits', help='Dataset: digits')
    parser.add_argument('--source', type = str, default='mnist', help='source dataset')
    parser.add_argument('--target', type = str, default='svhn', help='target dataset')
    parser.add_argument('--save_path', type = str, default='./checkpoint', help='path to save the checkpoint')
    parser.add_argument('--verbose', action = 'store_true', help='State the Variables and be verbose')
    parser.add_argument('--seed', type = int, default=0, help = 'random seeds')
    parser.add_argument('--forget', type=str, default='1,2', help='The class labels to forget')
    parser.add_argument('--device', type=str, default='cuda', help='device to train on')

    parser.add_argument('--source_lr', type=float, default=1e-2, help='learning rate for models')
    parser.add_argument('--lambda', type=float, default=0.1, help='lambda for entropy regularization')
    parser.add_argument('--batch', type = int, default=64, help ='batch size')
    parser.add_argument('--train_val_split', type=float, default=0.8, help='train val split for the datasets')
    parser.add_argument('--source_epochs', type=int, default=10, help='number of epochs to train on source model')
    parser.add_argument('--pretrain', action='store_true', help='pretrain the source model')

    parser.add_argument('--adv_train', action='store_true', help='train the adversarial model')
    parser.add_argument('--lambda_adv', type=float, default=1, help='lambda for adversarial training')
    parser.add_argument('--adv_lr', type=float, default=1e-2, help='learning rate for adversarial training')
    parser.add_argument('--adv_epochs', type=int, default=10, help='number of epochs to train on adversarial model')
    parser.add_argument('--adv_momentum', type=float, default=0.9, help='momentum for adversarial training')
    parser.add_argument('--adv_weight_decay', type=float, default=1e-4, help='weight decay for adversarial training')
    parser.add_argument('--adv_lr_decay', type=float, default=0.75, help='learning rate decay for adversarial training')
    parser.add_argument('--adv_lr_gamma', type=float, default=0.001, help='learning rate gamma for adversarial training')
    parser.add_argument('--trade_off', type=float, default=1, help='trade off for adversarial training')
    
    parser.add_argument('--ft_lr', type=float, default=1e-3, help='learning rate for fine tuning')
    parser.add_argument('--lambda_pseudo', type=float, default=1, help='lambda for the weight opf forgetting in pseudo')
    parser.add_argument('--algorithm', type=str, default='ewc', help='algorithm to use for unlearning')
    parser.add_argument('--fisher', action='store_true', help='use fisher information for unlearning')
    parser.add_argument('--lambda_ewc', type=float, default=1e-7, help='lambda for ewc loss')
    parser.add_argument('--num_forget', type=int, default=100, help='number of samples to forget')
    parser.add_argument('--lambda_fisher', type=float, default=1e-3, help='lambda for fisher loss')
    parser.add_argument('--test', type = bool, default=True, help='test trained models')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint for the adversarial training (not the pretraining)')
    parser.add_argument('--fine_tune_epochs', type=int, default=1, help='number of epochs to finetune the model')
    parser.add_argument('--lambda_rl', type=float, default=1e-3, help='lambda for the re-parametrized lagrangian')
    args  = parser.parse_args()
    
    if (not torch.cuda.is_available()):
        args.device = 'cpu'
        
    # args.device = device
    
    
    return args




    # parser.add_argument('--index', type = int, default=4, help ='index of the training loader, what to train on 4 = MNIST, 5 = SVHN')
    # parser.add_argument('--baseline', action='store_true', help='baseline values needed or not?')
    # parser.add_argument('--pretrain_baseline_sub', action='store_true', help='pretrain local model')
    
    # parser.add_argument('--pretrain_epochs', type = int, default=5, help = 'model training iterations for pretraining')
    
    # parser.add_argument('--ipc', type = int, default=50, help = 'sampled noisy images per class')
    # parser.add_argument('--model', type=str, default='ConvNet', help='model')

    # parser.add_argument('--init', type = str, default='normal', help='initialization method for noisy images')

    # parser.add_argument('--kd', type = bool, default=True, help='knowledge distillation')
    # parser.add_argument('--pretrain_baseline', action='store_true', help='pretrain local model') # I think the default should be true ; )
    # parser.add_argument('--dadv', action='store_true', help='domain adversarial training')


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)     
    torch.cuda.manual_seed_all(args.seed) 
    random.seed(args.seed)
    pass
    
