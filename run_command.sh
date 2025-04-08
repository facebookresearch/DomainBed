python -m domainbed.scripts.sweep launch\
       --data_dir=/mnt/x/DomainGeneralizationDatasets\
       --output_dir=./sweep/chin_dd\
       --command_launcher dummy\
       --algorithms CHIN\
       --datasets OfficeHome\
       --hparams '{"dim_C": 1024, "lambda_adv": 0.1, "lambda_recon": 0.01, "lr": 1e-4}'\
       --n_hparams 1\
       --n_trials 3\
       --single_test_envs


python -m domainbed.scripts.sweep delete_incomplete\
       --data_dir=/mnt/x/DomainGeneralizationDatasets\
       --output_dir=./sweep/chin\
       --command_launcher dummy\
       --algorithms CHIN\
       --datasets OfficeHome\
       --hparams '{"dim_C": 1024, "lambda_adv": 0.1, "lambda_recon": 0.01, "lr": 1e-4}'\
       --n_hparams 1\
       --n_trials 3

       python -m domainbed.scripts.collect_results\
       --input_dir=./sweep/chin

############ DAS ############
python -m domainbed.scripts.train \
    --data_dir /mnt/x/DomainGeneralizationDatasets \
    --algorithm DAS \
    --dataset OfficeHome \
    --test_env 0 \
    --hparams '{"lambda_sim":0.5}' \
    --trial_seed 42


python -m domainbed.scripts.sweep launch\
       --data_dir=/mnt/x/DomainGeneralizationDatasets\
       --output_dir=./sweep/das_cka\
       --command_launcher dummy\
       --algorithms DAS\
       --datasets OfficeHome\
       --hparams '{"lambda_sim":0.5}'\
       --n_hparams 1\
       --n_trials 3\
       --single_test_envs

python -m domainbed.scripts.sweep launch\
       --data_dir=/mnt/x/DomainGeneralizationDatasets\
       --output_dir=./sweep/das_mi\
       --command_launcher local\
       --algorithms DAS_MI\
       --datasets OfficeHome\
       --hparams '{"lambda_mi":0.5, "bandwidth":0.3}'\
       --n_hparams 1\
       --n_trials 3\
       --single_test_envs

       python -m domainbed.scripts.collect_results\
       --input_dir=./sweep/chin

python -m domainbed.scripts.train \
    --data_dir /mnt/x/DomainGeneralizationDatasets \
    --algorithm DAS_MI \
    --dataset OfficeHome \
    --test_env 0 \
    --hparams '{"lambda_mi":0.5, "bandwidth":0.3}' \
    --trial_seed 42