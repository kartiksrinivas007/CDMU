CUDA_VISIBLE_DEVICES=1 python3 main.py --wandb --pretrain --resume --dset office --source Art --target Clipart  --adv_train --adv_epochs 20 
# python3 main.py --dset digits --adv_epochs 20 --adv_lr 1e-5 --batch 64 --forget 2,3,4 --source mnist --target mnistm --verbose --resume --source_epochs 1 --source_lr 0.01 --seed 0 --device cuda:2


