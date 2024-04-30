import os 
import sys
import numpy as np


# give a log scale for lambda value in 1e-4 to 1e-5 and num forget logscale from 50 to 1000

lambda_ewcs = np.logspace(-8, 1, num=4)
lambda_pseudos = np.logspace(-4, 1, num=4)
num_forgets = np.logspace(1, 3, num=2)
num_forgets = [int(i) for i in num_forgets]
ft_epochs = np.linspace(5, 10, num=2)
ft_epochs = [int(i) for i in ft_epochs]

for lewc in lambda_ewcs:
    for num in num_forgets:
        for fte in ft_epochs:
            for lp in lambda_pseudos:
                os.system(f'CUDA_VISIBLE_DEVICES=1,2,3 python main.py --algorithm=pseudo --dset digits --adv_epochs 20 --adv_lr 1.5e-5 --batch 64 --forget 2,3,4 --source mnist --target mnistm --verbose --resume --source_epochs 1 --source_lr 0.01 --seed 0 --fine_tune_epochs {fte} --lambda_ewc {lewc} --lambda_pseudo {lp} --num_forget {num}')
