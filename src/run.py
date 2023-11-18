from EdgeCluster import EdgeCluster
import initial_clustering as ic
import preprocessing_windows as pw
from utils import draw_graph, get_label, summary, write_to_csv, move_data
import time
from itertools import product

def experiment_synthetic_data(num_dimentions=2, num_streams_initial=0, num_streams_windows=3, num_segments=10, batch_size=100, plots=True):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """

    list_of_windows = pw.syntethic(num_dimentions, num_streams = num_streams_windows, num_segments=num_segments, batch_size=batch_size)
    initial_clustering = ic.syntethic(num_dimentions, num_streams = num_streams_initial)
    if plots:
        draw_graph(df = initial_clustering['clustering'][0]['data'], 
                df_metrics = initial_clustering['clustering_metrics'],
                window_metrics={},
                filename='initial_clustering', 
                title=f'Initial clustering of {num_dimentions}-dim data', 
                xaxis_label='Feature 1', 
                yaxis_label='Feature 2', 
                num_dimentions=num_dimentions, 
                color_pallete='tab10',
                batch_size=batch_size,
                path=['..', 'results', 'syntethic', f'{num_dimentions}-dim', 'plots'])
        
    # keep all the clustering solutions
    # the latest one is the final one
    list_of_clustering_solutions = []

    offline_clustering = initial_clustering['clustering_metrics']
    for index, window in enumerate(list_of_windows['clustering_metrics']):
        print(f"Start processing a window {index}")
        clustering = EdgeCluster().fit(window, offline_clustering)
        move_data(initial_clustering, clustering, list_of_windows)
        print(f"The EdgeCluster completed for a window {index}")
        print(f"Start plotting a graph for a window {index}")
        if plots:
            draw_graph(df = initial_clustering['clustering'][0]['data'], 
                df_metrics = clustering['clustering'],
                window_metrics=[window],
                filename=f'clustering_window_{index + 1}_stream_{window["stream"]}_segment_{window["segment"]}_{get_label(clustering)}', 
                title=f'Clustering of {num_dimentions}-dim data for window {index + 1}', 
                xaxis_label='Feature 1', 
                yaxis_label='Feature 2', 
                num_dimentions=num_dimentions, 
                color_pallete='tab10', 
                batch_size=batch_size,
                path = ['..', 'results', 'syntethic', f'{num_dimentions}-dim', 'plots'])
        print(f"The graph for a window {index} was plotted")
        list_of_clustering_solutions.append(clustering)
        print(f"Summary for a window {index}")
        df = summary(clustering)
        print(f"Write a csv for a window {index}")
        write_to_csv(filename=f'clustering_window_{index + 1}_stream_{window["stream"]}_segment_{window["segment"]}_{get_label(clustering)}', 
                     data=df,
                     path = ['..', 'results', 'syntethic', f'{num_dimentions}-dim', 'tabular', f'{batch_size}'])
       
    write_to_csv(filename='final_clustering', 
                    data=initial_clustering['clustering'][0]['data'], 
                    path = ['..', 'results', 'syntethic', f'{num_dimentions}-dim', 'tabular', f'{batch_size}'])
    return list_of_clustering_solutions

def experiment_ampds2_data(num_segments: int, batch_size: int):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """
    hours = [1, 2, 3, 4, 6, 8]
    types = ['all', 'gas', 'water', 'weather', 'elec']

    # generate combinations
    combinations = product(hours, types)
    for hour, type in combinations:
        list_of_windows = pw.ampds(hour, type, num_segments, batch_size)
        initial_clustering = ic.ampds(hour, type)
        for w in list_of_windows:
            EdgeCluster().fit(w, initial_clustering)

if __name__ == '__main__':
    size_windows = [10, 25, 50, 75, 100, 250, 500, 750, 1000]   # number of samples
    start_time = time.time()
    for size in [10]:
        print(f"Starting size {size}")
        # 3-streams with 2-dimensional data
        experiment_synthetic_data(num_dimentions=2, num_streams_initial=0, num_streams_windows=3, batch_size=size)
    for size in size_windows:
        print(f"Starting size {size}") 
        # 12-streams with 8-dimensional data
        experiment_synthetic_data(num_dimentions=8, num_streams_initial=0, num_streams_windows=12, batch_size=size, plots=False)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # start_time = time.time()
    # experiment_ampds2_data(num_segments=10, batch_size=10)
    # print("--- %s seconds ---" % (time.time() - start_time))
