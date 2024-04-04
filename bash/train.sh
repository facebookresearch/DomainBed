#not wandb
# CUDA_VISIBLE_DEVICES=1 \
# python3 -m domainbed.scripts.train\
#        --data_dir=./domainbed/data/\
#        --algorithm CAG\
#        --dataset PACS\
#        --test_env 0\
#        --steps 5000\
#        --hparams_seed 0\
#        --trial_seed 0\
#        # --wandb

python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
       --algorithm CAG\
       --dataset ColoredMNIST\
       --test_env 0\
       --hparams_seed 12\
       --hparams '{"batch_size": 323, "cag_update": 1, "cagrad_c": 0.5,  "lr": 0.002779717688784654, "meta_lr": 0.8, "resnet_dropout": 0.5, "weight_decay": 0.0}'\
       --trial_seed 0\
       --seed 1028169927\
       --steps 5000\