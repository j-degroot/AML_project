# Sequential Model Algorithm Configuration (SMAC)

The installation is taken from the repository https://github.com/automl/SMAC3.  The file `get_optimization_data.py` is adopted from the repository minimal example.


## Installation

Create a new environment with python 3.9 and make sure swig is installed either on your system or
inside the environment. We demonstrate the installation via anaconda in the following:

Create and activate environment:
```
conda create -n SMAC python=3.9
conda activate SMAC
```

Install swig:
```
conda install gxx_linux-64 gcc_linux-64 swig
```

Install SMAC via PyPI:
```
pip install smac
```

Or alternatively, clone the environment:
```
git clone https://github.com/automl/SMAC3.git && cd SMAC3
pip install -r requirements.txt
pip install .
```

We refer to the [documention](https://automl.github.io/SMAC3) for further installation options.



## Miscellaneous

SMAC3 is developed by the [AutoML Groups of the Universities of Hannover and
Freiburg](http://www.automl.org/).

If you have found a bug, please report to [issues](https://github.com/automl/SMAC3/issues). Moreover, we are appreciating any kind of help.
Find our guidlines for contributing to this package [here](https://github.com/automl/SMAC3/blob/master/.github/CONTRIBUTING.md).

If you use SMAC in one of your research projects, please cite us:
```
@misc{lindauer2021smac3,
      title={SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization}, 
      author={Marius Lindauer and Katharina Eggensperger and Matthias Feurer and André Biedenkapp and Difan Deng and Carolin Benjamins and René Sass and Frank Hutter},
      year={2021},
      eprint={2109.09831},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Copyright (C) 2016-2021  [AutoML Group](http://www.automl.org/).
