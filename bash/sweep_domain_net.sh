python -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/\
       --output_dir=./train_output\
       --command_launcher local\
       --algorithms CAG1 \
       --datasets DomainNet \
       --n_hparams 5\
       --n_trials 3\
       --skip_confirmation\
       --wandb\