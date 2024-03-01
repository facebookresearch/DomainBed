import os, sys, shutil
import time

import wandb
from torch.utils.tensorboard import SummaryWriter
import pickle


class Logging:
    def __init__(self, args, hparam):
        self.__log = {}
        self.__epoch = 0

        if args.wandb:
            args.run_name = (f"{args.dataset}_{args.algorithm}"
                             f"_{args.task}-{args.steps}"
                             f"-{args.test_envs}"
                             f"__{int(time.time())}")

            self.__run = wandb.init(
                project=args.wandb_prj,
                entity=args.wandb_entity,
                config=args,
                name=args.run_name,
                force=True
            )

        if args.log:
            self.__writer = SummaryWriter(args.exp_dir)
            self.__writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

        self.__args = args

    def __call__(self, key, value):
        if key in self.__log:
            self.__log[key] += value
        else:
            self.__log[key] = value

    def __update_wandb(self):
        for log_key in self.__log_avg:
            self.__run.log({log_key: self.__log_avg[log_key]}, step=self.__epoch)

    def __update_board(self):
        for log_key in self.__log_avg:
            self.__writer.add_scalar(log_key, self.__log_avg[log_key], self.__epoch)

    def __reset_epoch(self):
        self.__log = {}

    def reset(self):
        self.__reset_epoch()
        self.__epoch = 0

    def step(self, epoch, test_len):
        self.__epoch = epoch

        self.__log_avg = {}
        for log_key in self.__log:
            if log_key.split("/")[0] in ['train']:
                self.__log_avg[log_key] = self.__log[log_key]
            else:
                self.__log_avg[log_key] = self.__log[log_key] / test_len


        if self.__args.wandb:
            self.__update_wandb()

        if self.__args.log:
            self.__update_board()

        self.__reset_epoch()

    def log(self, path, value, step):
        self.__writer.add_scalar(path, value, step)
        wandb.log({path: value}, step=step)

    def watch(self, model, num_train_batch):
        self.__run.watch(models=model, log='all', log_freq=num_train_batch, log_graph=True)
    
    def save_file(self, path):
        if self.__args.wandb:
            # Extract the filename from the path.
            filename = os.path.basename(path)
            # Copy the file to the current working directory.
            temp_path = os.path.join(os.getcwd(), filename)
            shutil.copy(path, temp_path)
            # Save the file with wandb.
            self.__run.save(filename)
            # Remove the copied file to clean up.
            os.remove(temp_path)
            print(filename,temp_path)

    @property
    def log(self):
        return self._Logging__log

    @property
    def log_avg(self):
        return self._Logging__log_avg

    @property
    def epoch(self):
        return self._Logging__epoch