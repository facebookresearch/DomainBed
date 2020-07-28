# Welcome to DomainBed

DomainBed is a PyTorch suite containing benchmark datasets and algorithms for domain generalization, as introduced in [In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434).

## Quick start

Download the datasets:

```sh
python -m domainbed.scripts.download \
       --data_dir=/my/datasets/path
```

Train a model:

```sh
python -m domainbed.scripts.train\
       --data_dir=/my/datasets/path\
       --algorithm ERM\
       --dataset RotatedMNIST
```

Launch a sweep:

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher
```

Here, `MyLauncher` is your cluster's command launcher, as implemented in `command_launchers.py`. At the time of writing, the entire sweep trains 50,000 models (all algorithms x all datasets x 3 independent trials x 20 random hyper-parameter choices). You can pass arguments to make the sweep smaller:

```sh
python -m domainbed.scripts.sweep launch\
       --data_dir=/my/datasets/path\
       --output_dir=/my/sweep/output/path\
       --command_launcher MyLauncher\
       --algorithms ERM DANN\
       --datasets RotatedMNIST VLCS\
       --n_hparams 5\
       --n_trials 1
```

After all jobs have either succeeded or failed, you can delete the data from failed jobs with ``python -m domainbed.scripts.sweep delete_incomplete`` and then re-launch them by running ``python -m domainbed.scripts.sweep launch`` again. Specify the same command-line arguments in all calls to `sweep` as you did the first time; this is how the sweep script knows which jobs were launched originally.

To view the results of your sweep:

````sh
python -m domainbed.scripts.collect_results\
       --input_dir=/my/sweep/output/path
````

## Implementing new algorithms and datasets

Check [``domainbed/algorithms.py``](domainbed/algorithms.py) and [``domainbed/datasets.py``](domainbed/datasets.py). Pull requests welcome.

## Running tests

DomainBed includes some unit tests and end-to-end tests. While not exhaustive, but they are a good sanity-check. To run the tests:

```sh
python -m unittest discover
```

By default, this only runs tests which don't depend on a dataset directory. To run those tests as well:

```sh
DATA_DIR=/my/datasets/path python -m unittest discover
```

## License

This source code is released under the MIT license, included [here](LICENSE). 
