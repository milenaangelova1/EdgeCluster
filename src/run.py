from EdgeCluster import EdgeCluster
import initial_clustering as ic
import preprocessing_windows as pw
from utils import draw_graph, get_label, summary, write_to_csv, move_data, preprocessing_final_dataset
from metrics import evalutation_report
import time
import pandas as pd

def experiment_synthetic_data(num_dimentions=2, stream_number=0, num_segments=10, batch_size=100, plots=True):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """
    list_of_windows = pw.synthetic(num_dimentions, stream_number=stream_number, num_segments=num_segments, batch_size=batch_size)
    initial_clustering = ic.synthetic(num_dimentions, stream_number=stream_number, size=batch_size)
    
    if plots:
        draw_graph(df_metrics = initial_clustering['clustering_metrics'],
                window_metrics={},
                filename='initial_clustering', 
                title=f'Initial clustering of {num_dimentions}-dim data', 
                xaxis_label='Feature 1', 
                yaxis_label='Feature 2', 
                batch_size=batch_size,
                path=['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'plots', f'stream {stream_number}'],
                initial_graph=True)
        
    # keep all the clustering solutions
    # the latest one is the final one
    list_of_clustering_solutions = []
    dfs = []
    for index, window in enumerate(list_of_windows['clustering_metrics']):
        print(f"Start processing a window {index}")
        clustering = EdgeCluster().fit(window, initial_clustering['clustering_metrics'])
        move_data(initial_clustering, clustering, list_of_windows)
        print(f"The EdgeCluster completed for a window {index}")
        print(f"Start plotting a graph for a window {index}")
        if plots:
            draw_graph(df_metrics = clustering,
                window_metrics=[window],
                filename=f'clustering_window_{index}_stream_{window["stream"]}_segment_{window["segment"]}_{get_label(clustering)}', 
                title=f'Clustering of {num_dimentions}-dim data for window {index}', 
                xaxis_label='Feature 1', 
                yaxis_label='Feature 2', 
                batch_size=batch_size,
                path = ['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'plots', f'stream {stream_number}'])
        print(f"The graph for a window {index} was plotted")
        list_of_clustering_solutions.append(clustering)
        print(f"Summary for a window {index}")
        df = summary(clustering)
        print(f"Write a csv for a window {index}")
        df['index'] = df.shape[0] * [index]
        dfs.append(df)
        write_to_csv(filename=f'clustering_window_{index}_stream_{window["stream"]}_segment_{window["segment"]}_{get_label(clustering)}', 
                     data=df,
                     path = ['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'tabular', f'stream {stream_number}', f'{batch_size}'])
    
    final_df = preprocessing_final_dataset(initial_clustering['clustering'])
    # final_df = final_df.drop_duplicates(final_df.columns[:-4], keep='first')
    write_to_csv(filename='final_clustering', 
                    data=final_df,
                    path = ['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'tabular', f'stream {stream_number}', f'{batch_size}'])
    
    monitoring_df = pd.concat(dfs, ignore_index=True, sort=False)
    write_to_csv(filename='final_minitoring', 
                    data=monitoring_df, 
                    path = ['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'tabular', f'stream {stream_number}', f'{batch_size}'])
    
    segments = final_df['segment'].unique()
    metrics = []
    for segment in segments:
        df = final_df[final_df['segment'] == segment]
        metrics_dict = evalutation_report(data=df[df.columns[:-4]], pred_labels=df['cluster'].values, true_labels=df['target'].values)
        metrics_df =  pd.DataFrame({
            'connectivity': [metrics_dict["connectivity"]],
            'F1': [metrics_dict["F1"]],
            'SI': [metrics_dict["SI"]],
            'homogeneity': [metrics_dict["homogeneity"]],
            "RI": [metrics_dict["RI"]],
            "ARI": [metrics_dict["ARI"]],
            "MI": [metrics_dict["MI"]],
            "NMI": [metrics_dict["NMI"]],
            "AMI": [metrics_dict["AMI"]],
            "CS": [metrics_dict["CS"]],
            "V": [metrics_dict["V"]],
            "FMI": [metrics_dict["FMI"]],
            "S": [metrics_dict["S"]],
            "DB": [metrics_dict["DB"]],
            'segment': [segment],
            'stream': [stream_number]
        })

        metrics.append(metrics_df)
    
    return list_of_clustering_solutions, pd.concat(metrics, ignore_index=True, sort=False)

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
        write_to_csv(filename=f'clustering_window_{index}_{hour}H_{type}_{get_label(clustering)}', 
                    data=df,
                    path = ['..', 'results', 'ampds', 'tabular', f'{batch_size}'])
       
        write_to_csv(filename=f'final_clustering_{hour}H_{type}', 
                        data=initial_clustering['clustering'][0]['data'], 
                        path = ['..', 'results', 'ampds', 'tabular', f'{batch_size}'])
    return list_of_clustering_solutions

if __name__ == '__main__':
    # Experiment with synthetic data
    size_windows = [3, 5, 10, 100, 500, 1000]   # number of samples
    start_time = time.time()
    
    # Experiment 3-streams with 2-dimensional data
    final_data_metrics = []
    for stream_num in range(3):
        all_metrics = []
        for size in size_windows:
            print(f"Starting size {size}")
            # 3-streams with 2-dimensional data
            _, metrics = experiment_synthetic_data(num_dimentions=2, stream_number=stream_num, num_segments=10, batch_size=size)
            metrics['size'] = metrics.shape[0] * [size]
            all_metrics.append(metrics)
        final_stream_metrics_df = pd.concat(all_metrics, ignore_index=True, sort=False)
        final_data_metrics.append(final_stream_metrics_df)
        write_to_csv(filename='metrics', 
                data=final_stream_metrics_df, 
                path = ['..', 'results', 'synthetic', '2-dim', 'tabular', f'stream {stream_num}', f'{size}'])
    
    final_data_metrics_df = pd.concat(final_data_metrics, ignore_index=True, sort=False)
    write_to_csv(filename='final_evalution_metrics', 
                data=final_data_metrics_df, 
                path = ['..', 'results', 'synthetic', '2-dim', 'tabular'])

    # Experiment 12-streams with 8-dimensional data
    # final_data_metrics = []  
    # for stream_num in range(12): 
    #     all_metrics = []
    #     for size in size_windows:
    #         print(f"Starting size {size}") 
    #         _, metrics = experiment_synthetic_data(num_dimentions=8, stream_number=stream_num, num_segments=10, batch_size=size, plots=False)
    #         metrics['size'] = metrics.shape[0] * [size]
    #         all_metrics.append(metrics)
    #     final_stream_metrics_df = pd.concat(all_metrics, ignore_index=True, sort=False)
    #     final_data_metrics.append(final_stream_metrics_df)
    #     write_to_csv(filename='metrics', 
    #             data=pd.concat(all_metrics, ignore_index=True, sort=False), 
    #             path = ['..', 'results', 'synthetic', '8-dim', 'tabular', f'stream {stream_num}', f'{size}'])
        
    # final_data_metrics_df = pd.concat(final_data_metrics, ignore_index=True, sort=False)
    # write_to_csv(filename='final_evalution_metrics', 
    #             data=final_data_metrics_df, 
    #             path = ['..', 'results', 'synthetic', '8-dim', 'tabular'])
        
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
