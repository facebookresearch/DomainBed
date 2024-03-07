# cd ..
# delete_incomplete
# RotatedMNIST ColoredMNIST
python -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/MNIST/\
       --output_dir=./train_output\
       --command_launcher local\
       --algorithms ERM GradBase CAG Fish \
       --datasets RotatedMNIST \
       --n_hparams 1\
       --n_trials 1\
       --skip_confirmation\
       --wandb\