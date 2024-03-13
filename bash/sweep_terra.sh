python -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/\
       --output_dir=./train_output\
       --command_launcher local\
       --algorithms CAG1 \
       --datasets TerraIncognita \
       --n_hparams 1\
       --n_trials 1\
       --skip_confirmation\
       --wandb\