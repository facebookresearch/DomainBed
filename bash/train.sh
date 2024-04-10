#not wandb
# CUDA_VISIBLE_DEVICES=1 \
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
       --algorithm Fishr\
       --dataset ColoredMNIST\
       --test_env 0\
       --steps 5000\
       --hparams_seed 0\
       --trial_seed 0\
       # --wandb

