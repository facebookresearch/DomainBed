#not wandb

python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
       --algorithm ERM\
       --dataset PACS\
       --test_env 0\
       --steps 301\