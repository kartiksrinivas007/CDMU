import os
import numpy as np

lambda_rls = np.logspace(-8, -1, num=5)
# lambda_pseudos = np.logspace(-4, 1, num=5)
ft_lrs = np.logspace(-7, -2, num=5)
num_forgets = np.logspace(1, 3, num=3)
num_forgets = [int(i) for i in num_forgets]
ft_epochs = np.linspace(1, 10, num=3)
ft_epochs = [int(i) for i in ft_epochs]

for ft_lr in ft_lrs:
    for num in num_forgets:
        for fte in ft_epochs:
            for lam_rl in lambda_rls:
                os.system(f'CUDA_VISIBLE_DEVICES=1,2,3 python main.py --algorithm=rl --dset digits --adv_epochs 20 --adv_lr 1.5e-5 --batch 64 --forget 2,3,4 --source mnist --target usps --verbose --resume --source_epochs 1 --source_lr 0.01 --seed 0 --fine_tune_epochs {fte} --lambda_rl {lam_rl} --ft_lr {ft_lr} --num_forget {num}')