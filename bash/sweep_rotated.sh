# cd ..
# delete_incomplete
# RotatedMNIST ColoredMNIST
# ERM GradBase CAG Fish
CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.sweep launch\
       --data_dir=./domainbed/data/MNIST/\
       --output_dir=./MnistTest\
       --command_launcher local\
       --algorithms CAG \
       --datasets RotatedMNIST\
       --n_hparams 30\
       --n_trials 3\
       --skip_confirmation\
       --single_test_envs\
