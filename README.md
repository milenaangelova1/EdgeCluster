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

## Experiments
### Preprocessing of the both datasets (synthetic and AMPDS2)
#### Synthetic data
For 2-dim dataset:
    The data is separated into 3 streams (0, 1, and 2), and each stream consists of 10 data segments.
    Each data segment has 1000 samples inside. The data points are between 0 and 1 and have cluster labels.
    Preprocessing: For the initial clustering, I took all files that are under stream 0 and combined them in one dataset. According to this, we received the following clusters and their number of samples: \\
    Cluster 0: 2026 \\
    Cluster 1: 2277 \\
    Cluster 2: 3341 \\
    Cluster 3: 1540 \\
    Cluster 4: 816 \\

    The result of the initial clustering is that some clusters are disjoined, but not all. Some are overlapped. All other data segments are represented as a list of windows with a fixed size.

- For 8-dim dataset:
    The data is separated into 12 streams, and each stream consists of 10 data segments.
    Each data segment has 1000 samples inside. The data points are between 0 and 1; they also have cluster labels.
    Preprocessing: for the initial clustering, I took all files that are under stream 0 and combined them in one dataset. According to this, we received the following clusters and their number of samples: \\
    Cluster 0:  \\
    Cluster 1:  \\
    Cluster 2:  \\
    Cluster 3:  \\
    Cluster 4:  \\

Current version for this experiment: tag 1.0.0
All the results are archived.

#### Discussion on 29.11.2023
The initial clustering must not have clusters that are overlapped. The decision is: the stream 0 segment 0 will be the initial clustering, and all other data segments will be windows with a fixed size. \\
Represent the clusters as rectangles in the plots.
Validate the final clustering solution.
Implement validation metrics. (Take them from the Christian's repository)
