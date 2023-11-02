from EdgeCluster import EdgeCluster
import initial_clustering as ic
import preprocessing_windows as pw
from utils import draw_graph
from utils import get_label

def experiment_synthetic_data(num_dimentions=2, num_streams_initial=0, num_streams_windows=3, num_windows=10):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """

    list_of_windows = pw.syntethic(num_dimentions, num_streams = num_streams_windows,  num_windows = num_windows)
    initial_clustering = ic.syntethic(num_dimentions, num_streams = num_streams_initial)
    draw_graph(df = initial_clustering['clustering'][0]['data'], 
               df_metrics = initial_clustering['clustering_metrics'],
               window_metrics={},
               filename='initial_clustering', 
               title=f'Initial clustering of {num_dimentions}-dim data', 
               xaxis_label='Feature 1', 
               yaxis_label='Feature 2', 
               num_dimentions=num_dimentions, 
               color_pallete='tab10')
    
    # keep all the clustering solutions
    # the latest one is the final one
    list_of_clustering_solutions = []

    offline_clustering = initial_clustering['clustering_metrics']
    for index, window in enumerate(list_of_windows['clustering_metrics']):
        clustering = EdgeCluster().fit(window, offline_clustering)
        draw_graph(df = initial_clustering['clustering'][0]['data'], 
               df_metrics = clustering['clustering'],
               window_metrics=[window],
               filename=f'clustering_window_{index + 1}_stream_{window["stream"]}_segment_{window["segment"]}_{get_label(clustering)}', 
               title=f'Clustering of {num_dimentions}-dim data for window {index + 1}', 
               xaxis_label='Feature 1', 
               yaxis_label='Feature 2', 
               num_dimentions=num_dimentions, 
               color_pallete='tab10')
        list_of_clustering_solutions.append(clustering)
    return list_of_clustering_solutions

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
    # experiment_synthetic_data(num_dimentions=8, num_streams_initial=0, num_streams_windows=12, num_windows=10)

    # experiment_ampds2_data(num_dimentions=2, num_streams=3, num_windows=10)