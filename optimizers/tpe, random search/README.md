

# Installation

```bash
# create a new environment
# install dependencies
pip install -r requirements.txt
```

# Running Experiments

## TPE on logistic regression on MNIST data
Performs TPE hyperparameter optimization of logistic regression on original MNIST dataset. Gathers the data of performance for each hyperparameters seting. 

```bash
# Run TPE HPO on logistic regression on MNIST data
python -m main_tpe_logreg
```

## Random search on logistic regression on MNIST data
Performs random search of hyperparameters of logistic regression on original MNIST dataset. Gathers the data of performance for each hyperparameters seting. 

```bash
# Run TPE HPO on logistic regression on MNIST data
python -m main_random_logreg
```

## TPE on surrogate benchmark model
Performs TPE hyperparameter optimization with trained surrogate benchmark model - which is Random Forest regressor. Random Forest Regressor is trained once at the beginning and prediction of loss from that model is used as objective function for TPE. Gathers loss per evaluation. 

```bash
# Run TPE HPO on logistic regression on MNIST data
python -m main_tpe_forest
```

## Random search on logistic regression on MNIST data
Performs random search of hyperparameters of Random Forest Regresor. We used random search to optimize hyperparameters and considered 100 samples over the stated hyperparameters; we trained the model on 50% of the data, chose the best configuration based on its performance on the other 50%. Returns the best founded parameters.

```bash
# Run TPE HPO on logistic regression on MNIST data
python -m main_random_logreg
```


