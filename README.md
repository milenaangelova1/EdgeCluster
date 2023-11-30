# EdgeCluster

## Create a virtual environment
```
python -m venv .venv
```

## Activate the environemnt

### for windows users
```
. .venv/Scripts/activate
```

### for mac and linux users
```
. .venv/bin/activate
```

## Install required python packages
```
pip install -r requirements.txt
```

## Start experiments
```
python run.py
```
The run file will execute all experiments.
The first expriment is with the synthetic data. 
Starts with 2-dim dataset and then continues with other one - 8-dim.

The second experiment is with the AMPDS2 dataset.