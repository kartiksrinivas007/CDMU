import os 
import sys
import numpy as np


# give a log scale for lambda value in 1e-4 to 1e-5 and num forget logscale from 50 to 1000

lambdas = np.logspace(-4, -5, num=10)
num_forgets = np.logspace(1.7, 3, num=10)
num_forgets = [int(i) for i in num_forgets]
ft_epochs = np.linspace(1, 10, num=3)
ft_epochs = [int(i) for i in ft_epochs]


for lambd in lambdas:
    for forget in num_forgets:
        # breakpoint()
        os.system('CUDA_VISIBLE_DEVICES=1,2,3 python main.py --algorithm=newton --dset digits --adv_epochs 20 --adv_lr 1.5e-5 --batch 64 --forget 2,3,4 --source mnist --target usps --verbose --resume --source_epochs 1 --source_lr 0.01 --seed 0 --lambda_fisher {} --num_forget {}'.format(lambd, forget))