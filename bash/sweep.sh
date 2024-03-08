# cd ..
# delete_incomplete
# RotatedMNIST ColoredMNIST
CUDA_VISIBLE_DEVICES=1 \ 
python -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/MNIST/\
       --output_dir=./train_output\
       --command_launcher local\
       --algorithms ERM GradBase CAG Fish \
       --datasets ColoredMNIST \
       --n_hparams 1\
       --n_trials 3\
       --skip_confirmation\
       --wandb\
