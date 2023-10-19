from utils import get_data_windows, get_initial_clustering
import EdgeCluster

def experiment_synthetic_data():
    """
    Experiment: runs Edge Cluster over synthetic data.
    """

    list_of_windows = get_data_windows()
    C = get_initial_clustering()
    for w in list_of_windows:
        EdgeCluster().fit(w, C)

def experiment_ampds2_data():
    """
    Experiment: runs Edge Cluster over synthetic data.
    """

    list_of_windows = get_data_windows()
    C = get_initial_clustering()
    for w in list_of_windows:
        EdgeCluster().fit(w, C)

if __name__ == '__main__':
    experiment_synthetic_data()
    experiment_ampds2_data()