import EdgeCluster
from initial_clustering import syntethic, ampds
from preprocessing_windows import syntethic, ampds

def experiment_synthetic_data():
    """
    Experiment: runs Edge Cluster over synthetic data.
    """

    list_of_windows = syntethic()
    initial_clustering = syntethic()
    for window in list_of_windows:
        EdgeCluster().fit(window, initial_clustering)

def experiment_ampds2_data():
    """
    Experiment: runs Edge Cluster over synthetic data.
    """

    list_of_windows = ampds()
    C = ampds()
    for w in list_of_windows:
        EdgeCluster().fit(w, C)

if __name__ == '__main__':
    experiment_synthetic_data()
    experiment_ampds2_data()