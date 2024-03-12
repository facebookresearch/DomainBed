#not wandb
CUDA_VISIBLE_DEVICES=1 \
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
       --algorithm CAG\
       --dataset PACS\
       --test_env 0\
       --steps 2\
       # --hparams_seed 1\
       # --trial_seed 1\