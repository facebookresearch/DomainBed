python3 -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/MNIST/\
       --output_dir=./train_output1\
       --command_launcher dummy\
       --algorithms CAG1 \
       --datasets RotatedMNIST\
       --n_hparams 1\
       --n_trials 1\
       --skip_confirmation\
