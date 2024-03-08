CUDA_VISIBLE_DEVICES=1 \
python -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/\
       --output_dir=./train_output\
       --command_launcher local\
       --algorithms ERM GradBase CAG Fish \
       --datasets OfficeHome \
       --n_hparams 1\
       --n_trials 1\
       --skip_confirmation\
       --wandb\