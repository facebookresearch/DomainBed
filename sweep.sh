#!/bin/bash

# local variables
outputdir=/pub2/podg
datadir=/pub2/data

jobid=$(date +"%y%m%d%H%M%S")
echo starting ${jobid}
mkdir ${outputdir}/${jobid}
CUDA_VISIBLE_DEVICES=2 python -m domainbed.scripts.sweep launch\
       --data_dir=${datadir} \
       --output_dir=${outputdir}/${jobid} \
       --command_launcher local \
       --overlap 66 \
       --steps 2100 \
       --single_test_envs \
       --algorithms ERM \
       --datasets VLCS \
       --n_hparams 5\
       --n_trials 3
