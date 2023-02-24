#!/bin/bash

# local variables
outputdir=/pub2/podg
datadir=/pub2/data
algorithm=ERM
n_hparams=5
steps=5001
trial=3

jobid=$(date +"%y%m%d%H%M%S")
overlap=100
datasets=VLCS
current_output_dir=${outputdir}/${jobid}_${algorithm}_${datasets}_o${overlap}_h${n_hparams}_s${steps}_t${trial}
echo starting ${current_output_dir}
mkdir current_output_dir

CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.sweep launch\
       --data_dir=${datadir} \
       --output_dir=${current_output_dir} \
       --command_launcher local \
       --overlap ${overlap} \
       --steps ${steps} \
       --single_test_envs \
       --algorithms ${algorithm} \
       --datasets ${datasets} \
       --n_hparams ${n_hparams} \
       --n_trials ${trial}
