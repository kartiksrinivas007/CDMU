python3 main.py --algorithm=ewc --dset digits --adv_epochs 20 --adv_lr 1.5e-5 --batch 64 --forget 2,3,4 --source mnist --target mnistm --verbose --resume --source_epochs 1 --source_lr 0.01 --seed 0 --fisher --ft_lr 1e-8 --lambda_ewc 1e-1 --lambda_pseudo 1e-1 --fine_tune_epochs 10
# python3 main.py --dset digits --adv_epochs 20 --adv_lr 1e-5 --batch 64 --forget 2,3,4 --source mnist --target mnistm --verbose --resume --source_epochs 1 --source_lr 0.01 --seed 0 --device cuda:2


