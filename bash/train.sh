#not wandb
# CUDA_VISIBLE_DEVICES=1 \
python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
<<<<<<< HEAD
       --algorithm CAG1\
       --dataset DomainNet\
       --test_env 0\
       --steps 2\
       --hparams_seed 0\
       --trial_seed 0\
=======
       --algorithm CAG\
       --dataset PACS\
       --test_env 0\
       --steps 2\
       # --hparams_seed 1\
       # --trial_seed 1\
>>>>>>> effba743229bcf367223ed008cb92247b541f99d
