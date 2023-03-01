#!/bin/bash

# local variables
outputdir=/pub2/podg
datadir=/pub2/data
datasets=PACS
n_hparams=5
steps=5001
trial=3
gpu=0
overlap=0

jobid=$(date +"%y%m%d%H%M%S")
algorithm=ERM
current_output_dir=${outputdir}/${overlap}/${algorithm}_${datasets}_o${overlap}_h${n_hparams}_s${steps}_t${trial}_${jobid}
echo starting ${current_output_dir}
mkdir -p current_output_dir

echo 'y' |CUDA_VISIBLE_DEVICES=${gpu} python -m domainbed.scripts.sweep launch\
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

