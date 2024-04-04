CUDA_VISIBLE_DEVICES=3 python3 -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/MNIST/\
       --output_dir=./MnistTest\
       --command_launcher local\
       --algorithms CAG \
       --datasets ColoredMNIST\
       --n_hparams 30\
       --n_trials 3\
       --skip_confirmation\
       --single_test_envs\