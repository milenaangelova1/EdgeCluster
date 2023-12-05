from EdgeCluster import EdgeCluster
import initial_clustering as ic
import preprocessing_windows as pw
from utils import draw_graph, get_label, summary, write_to_csv, move_data
import time

def experiment_synthetic_data(num_dimentions=2, stream_number=0, num_segments=10, batch_size=100, plots=True):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """

    list_of_windows = pw.synthetic(num_dimentions, stream_number=stream_number, num_segments=num_segments, batch_size=batch_size)
    initial_clustering = ic.synthetic(num_dimentions, stream_number=stream_number)
    if plots:
        draw_graph(df_metrics = initial_clustering['clustering_metrics'],
                window_metrics={},
                filename='initial_clustering', 
                title=f'Initial clustering of {num_dimentions}-dim data', 
                xaxis_label='Feature 1', 
                yaxis_label='Feature 2', 
                batch_size=batch_size,
                path=['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'plots', f'stream {stream_number}'])
        
    # keep all the clustering solutions
    # the latest one is the final one
    list_of_clustering_solutions = []

    for index, window in enumerate(list_of_windows['clustering_metrics']):
        print(f"Start processing a window {index}")
        clustering = EdgeCluster().fit(window, initial_clustering['clustering_metrics'])
        move_data(initial_clustering, clustering, list_of_windows)
        print(f"The EdgeCluster completed for a window {index}")
        print(f"Start plotting a graph for a window {index}")
        if plots:
            draw_graph(df_metrics = clustering['clustering'],
                window_metrics=[window],
                filename=f'clustering_window_{index + 1}_stream_{window["stream"]}_segment_{window["segment"]}_{get_label(clustering)}', 
                title=f'Clustering of {num_dimentions}-dim data for window {index + 1}', 
                xaxis_label='Feature 1', 
                yaxis_label='Feature 2', 
                batch_size=batch_size,
                path = ['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'plots', f'stream {stream_number}'])
        print(f"The graph for a window {index} was plotted")
        list_of_clustering_solutions.append(clustering)
        print(f"Summary for a window {index}")
        df = summary(clustering)
        print(f"Write a csv for a window {index}")
        write_to_csv(filename=f'clustering_window_{index + 1}_stream_{window["stream"]}_segment_{window["segment"]}_{get_label(clustering)}', 
                     data=df,
                     path = ['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'tabular', f'stream {stream_number}', f'{batch_size}'])
       
    write_to_csv(filename='final_clustering', 
                    data=initial_clustering['clustering'][0]['data'], 
                    path = ['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'tabular', f'stream {stream_number}', f'{batch_size}'])
    return list_of_clustering_solutions

def experiment_ampds2_data(hour, type, num_segments: int, batch_size: int):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """
    list_of_clustering_solutions = []
    list_of_windows = pw.ampds(hour, type, num_segments, batch_size)
    initial_clustering = ic.ampds(hour, type)
    for index, w in enumerate(list_of_windows['clustering_metrics']):
        clustering = EdgeCluster().fit(w, initial_clustering['clustering_metrics'])
        move_data(initial_clustering, clustering, list_of_windows)
        print(f"The EdgeCluster completed for a window {index}")
        list_of_clustering_solutions.append(clustering)
        print(f"Summary for a window {index}")
        df = summary(clustering)
        print(f"Write a csv for a window {index}")
        write_to_csv(filename=f'clustering_window_{index + 1}_{hour}H_{type}_{get_label(clustering)}', 
                    data=df,
                    path = ['..', 'results', 'ampds', 'tabular', f'{batch_size}'])
       
        write_to_csv(filename=f'final_clustering_{hour}H_{type}', 
                        data=initial_clustering['clustering'][0]['data'], 
                        path = ['..', 'results', 'ampds', 'tabular', f'{batch_size}'])
    return list_of_clustering_solutions

if __name__ == '__main__':
    # Experiment with synthetic data
    size_windows = [3, 5, 10]   # number of samples
    start_time = time.time()
    
    for stream_num in range(3):
        for size in size_windows:
            print(f"Starting size {size}")
            # 3-streams with 2-dimensional data
            experiment_synthetic_data(num_dimentions=2, stream_number=stream_num, num_segments=10, batch_size=size)
    # for stream_num in range(12):
    #     for size in size_windows:
    #         print(f"Starting size {size}") 
    #         # 12-streams with 8-dimensional data
    #         experiment_synthetic_data(num_dimentions=8, stream_number=stream_num, num_segments=10, batch_size=size, plots=False)
    # print("--- %s seconds ---" % (time.time() - start_time))
    
    # # Experiment with AMPDS2 dataset
    # start_time = time.time()
    # hours = [1, 2, 3, 4, 6, 8]
    # types = ['all', 'gas', 'water', 'weather', 'elec']

    # # generate combinations
    # combinations = product(hours, types)
    # for hour, type in combinations:
    #     experiment_ampds2_data(hour, type, num_segments=10, batch_size=None)
    # print("--- %s seconds ---" % (time.time() - start_time))
