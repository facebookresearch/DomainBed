#not wandb
CUDA_VISIBLE_DEVICES=1 \
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
       --algorithm CAG\
       --dataset OfficeHome\
       --test_env 0\
       --steps 301\
       # --hparams_seed 1\
       # --trial_seed 1\