# CUDA_VISIBLE_DEVICES=1\ 
python3 -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/MNIST/\
       --output_dir=./train_output\
       --command_launcher local\
       --algorithms CAG \
       --datasets ColoredMNIST\
       --n_hparams 20\
       --n_trials 3\
       --skip_confirmation\
       --wandb\
       --single_test_envs\