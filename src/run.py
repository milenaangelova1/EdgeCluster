from EdgeCluster import EdgeCluster
import initial_clustering as ic
import preprocessing_windows as pw

def experiment_synthetic_data(num_dimentions=2, num_streams_initial=0, num_streams_windows=3, num_windows=10):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """

    list_of_windows = pw.syntethic(num_dimentions, num_streams = num_streams_windows,  num_windows = num_windows)
    initial_clustering = ic.syntethic(num_dimentions, num_streams = num_streams_initial)
    clustering_all_windows = []

    for window in list_of_windows:
        clustering_all_windows.append(EdgeCluster().fit(window, initial_clustering))

def experiment_ampds2_data(num_dimentions=2, num_streams=3, num_windows=10):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """
    list_of_windows = pw.ampds(num_dimentions, num_streams, num_windows)
    initial_clustering = ic.ampds(num_dimentions, num_streams)
    for w in list_of_windows:
        EdgeCluster().fit(w, initial_clustering)

if __name__ == '__main__':
    # 3-streams with 2-dimensional data
    experiment_synthetic_data(num_dimentions=2, num_streams_initial=0, num_streams_windows=3, num_windows=10)
    # 12-streams with 8-dimensional data
    experiment_synthetic_data(num_dimentions=8, num_streams=12, num_windows=10)

    # experiment_ampds2_data(num_dimentions=2, num_streams=3, num_windows=10)