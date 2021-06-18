# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import os
import subprocess
from multiprocessing import Pool
import time
import torch

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()
        
def slurm_launcher(commands):
    """
    Parallel job launcher for computationnal cluster using SLURM workload manager.
    An example of SBATCH options:

        #!/bin/bash
        #SBATCH --job-name=<job_name>
        #SBATCH --output=<job_name>.out
        #SBATCH --error=<job_name>_error.out
        #SBATCH --ntasks=4
        #SBATCH --cpus-per-task=8
        #SBATCH --gres=gpu:4
        #SBATCH --time=1-00:00:00
        #SBATCH --mem=81Gb

    Note: --cpus-per-task should match the N_WORKERS defined in datasets.py (default 8)
    Note: there should be equal number of --ntasks and --gres
    """

    with Pool(processes=int(os.environ["SLURM_NTASKS"])) as pool:

        processes = []
        for command in commands:
            process = pool.apply_async(
                subprocess.run, 
                [f'srun --ntasks=1 --cpus-per-task={os.environ["SLURM_CPUS_PER_TASK"]} --mem=20G --gres=gpu:1 --exclusive {command}'], 
                {"shell": True}
                )
            processes.append(process)
            time.sleep(10)

        for i, process in enumerate(processes):
            process.wait()
            print("//////////////////////////////")
            print("//// Completed ", i , " / ", len(commands), "////")
            print("//////////////////////////////")


REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'slurm_launcher': slurm_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
