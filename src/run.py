from EdgeCluster import EdgeCluster
import initial_clustering as ic
import preprocessing_windows as pw
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import (draw_graph, 
                   get_label, metrics_by_segments, 
                   summary, update_segments_dict, 
                   write_to_csv, 
                   move_data, 
                   preprocessing_final_dataset)
from metrics import evalutation_report
import time
import pandas as pd

def experiment_s1_data(num_segments, batch_size, type):
    """
    Experiment: runs Edge Cluster over S1 data.
    """
    list_of_windows = pw.s1(num_segments=num_segments, batch_size=batch_size, type=type.split("_")[0])
    initial_clustering = ic.s1(size=batch_size, type=type.split("_")[0])

    draw_graph(df_metrics = initial_clustering['clustering_metrics'],
            window_metrics={},
            filename='initial_clustering', 
            title='Initial clustering of S1 data', 
            xaxis_label='Feature 1', 
            yaxis_label='Feature 2', 
            batch_size=batch_size,
            path=['..', 'results', 's1', f'{type}', 'plots'],
            initial_graph=True)
    
    # keep all the clustering solutions
    # the latest one is the final one
    list_of_clustering_solutions = []
    dfs = []
    segments = {
        1: [],
        2: [],
        3: [],
        4: []
    }
    for index, window in enumerate(list_of_windows['clustering_metrics']):
        print(f"Start processing a window {index}")
        clustering = EdgeCluster().fit(window, initial_clustering['clustering_metrics'])
        move_data(initial_clustering, clustering, list_of_windows, index)
        update_segments_dict(segments, window['segment'], initial_clustering)
        print(f"The EdgeCluster completed for a window {index}")
        print(f"Start plotting a graph for a window {index}")
        draw_graph(df_metrics = clustering,
            window_metrics=[window],
            filename=f'clustering_window_{index}_segment_{window["segment"]}_{get_label(clustering)}', 
            title=f'Clustering of S1 data for window {index}', 
            xaxis_label='Feature 1', 
            yaxis_label='Feature 2', 
            batch_size=batch_size,
            path = ['..', 'results', 's1', f'{type}', 'plots'])
        
        d = initial_clustering['clustering'][0]['data']
        palette = sns.color_palette('hls', n_colors=len(set(d['cluster'])))
    
        sns.scatterplot(x=d['x'], y=d['y'], hue=d['cluster'], palette=palette).set(title=f"Window {index}")
        plt.xticks(np.arange(0, 1.1, 0.1))  
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.savefig(os.path.join('results', 's1', f'{type}', 'plots', f'{batch_size}', f'scatter_clustering_window_{index}_segment_{window["segment"]}_{get_label(clustering)}.png'))
        
        print(f"The graph for a window {index} was plotted")
        list_of_clustering_solutions.append(clustering)
        print(f"Summary for a window {index}")
        df = summary(clustering)
        print(f"Write a csv for a window {index}")
        df['index'] = df.shape[0] * [index]
        dfs.append(df)
        write_to_csv(filename=f'clustering_window_{index}_segment_{window["segment"]}_{get_label(clustering)}', 
                     data=df,
                     path = ['..', 'results', 's1', f'{type}', 'tabular', f'{batch_size}'])
    
    # write to csv - final clustering
    final_df = preprocessing_final_dataset(initial_clustering['clustering'])
    write_to_csv(filename='final_clustering', 
                    data=final_df,
                    path = ['..', 'results', 's1', f'{type}', 'tabular', f'{batch_size}'])
    
    # write to csv - final monitoring
    monitoring_df = pd.concat(dfs, ignore_index=True, sort=False)
    write_to_csv(filename='final_minitoring', 
                    data=monitoring_df, 
                    path = ['..', 'results', 's1', f'{type}', 'tabular', f'{batch_size}'])
    
    # calculate evaluation metrics
    metrics, metrics_without = metrics_by_segments(segments, batch_size, 
                                                   type=type, 
                                                   path=['..', 'results', 's1', f'{type}', 'tabular', f'{batch_size}'],
                                                   true_labels=True)
    
    return list_of_clustering_solutions, pd.concat(metrics, ignore_index=True, sort=False), pd.concat(metrics_without, ignore_index=True, sort=False)

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

def experiment_ampds2_data(hour, type, num_segments: int, batch_size: int, metric: str):
    """
    Experiment: runs Edge Cluster over synthetic data.
    """
    initial_clustering = ic.ampds(hour, type, size=batch_size)
    list_of_windows = pw.ampds(hour, type, num_segments, batch_size)
    
    list_of_clustering_solutions = []
    dfs = []
    segments = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [],
        11: []
    }

    for index, w in enumerate(list_of_windows['clustering_metrics']):
        print(f"Start processing a window {index}")
        clustering = EdgeCluster().fit(w, initial_clustering['clustering_metrics'])
        move_data(initial_clustering, clustering, list_of_windows, index)
        update_segments_dict(segments, w['segment'], initial_clustering)
        print(f"The EdgeCluster completed for a window {index}")
        list_of_clustering_solutions.append(clustering)
        print(f"Summary for a window {index}")
        df = summary(clustering)
        print(f"Write a csv for a window {index}")
        df['index'] = df.shape[0] * [index]
        dfs.append(df)
        write_to_csv(filename=f'clustering_window_{index}_{hour}H_{type}_{get_label(clustering)}', 
                    data=df,
                    path = ['..', 'results', 'ampds', f'{type}', f'{batch_size}', 'tabular'])
       
        # write to csv - final clustering
    final_df = preprocessing_final_dataset(initial_clustering['clustering'])
    write_to_csv(filename='final_clustering', 
                    data=final_df,
                    path = ['..', 'results', 'ampds', f'{type}', f'{batch_size}','tabular'])
    
    # write to csv - final monitoring
    monitoring_df = pd.concat(dfs, ignore_index=True, sort=False)
    write_to_csv(filename='final_minitoring', 
                    data=monitoring_df, 
                    path = ['..', 'results', 'ampds', f'{type}', f'{batch_size}', 'tabular'])
    # calculate evaluation metrics
    metrics, metrics_without = metrics_by_segments(segments, batch_size, 
                                                   type='original', 
                                                   path=['..', 'results', 'ampds', f'{type}', f'{batch_size}', 'tabular'], 
                                                   true_labels=False,
                                                   metric=metric)
    
    return list_of_clustering_solutions, pd.concat(metrics, ignore_index=True, sort=False), pd.concat(metrics_without, ignore_index=True, sort=False)

def experiment1(type='continous'):
    size_windows = [3, 4, 6, 8, 12,24,30,32,48]   # number of samples in each window
    # size_windows = [24]
    start_time = time.time()

    # Experiment with S1 data
    final_data_metrics = []
    final_data_metrics_without = []
    for size in size_windows:
        print(f"Starting size {size}")
        _, metrics, metrics_without = experiment_s1_data(num_segments=4, batch_size=size, type=type)
        metrics['size'] = metrics.shape[0] * [size]
        metrics_without['size'] = metrics_without.shape[0] * [size]
        final_data_metrics.append(metrics)
        final_data_metrics_without.append(metrics_without)
        write_to_csv(filename='metrics', 
                data=metrics, 
                path = ['..', 'results', 's1', f'{type}', 'tabular', f'{size}'])
        write_to_csv(filename='metrics_without_deviation_and_matching', 
                data=metrics_without, 
                path = ['..', 'results', 's1', f'{type}', 'tabular', f'{size}'])
    
    final_data_metrics_df = pd.concat(final_data_metrics, ignore_index=True, sort=False)
    write_to_csv(filename='final_evalution_metrics', 
                data=final_data_metrics_df, 
                path = ['..', 'results', 's1', f'{type}', 'tabular'])
    
    final_data_metrics_df = pd.concat(final_data_metrics_without, ignore_index=True, sort=False)
    write_to_csv(filename='final_evalution_metrics_without_deviation_and_matching', 
                data=final_data_metrics_df, 
                path = ['..', 'results', 's1', f'{type}', 'tabular'])
    print("--- %s seconds ---" % (time.time() - start_time))

def experiment2():
    # Experiment with AMPDS2 dataset
    start_time = time.time()
    
    types = ['gas', 'water', 'elec', 'weather']
    # types = ['gas']
    hours = [6, 8, 4, 4]
    # hours = [6]
    size_windows = [3, 5, 7] # daily profiles
    # size_windows = [7]
    metrics = ['euclidean', 'euclidean', 'euclidean', 'canberra']

    final_data_metrics = []
    final_data_metrics_without = []
    for hour, type, metric in zip(hours, types, metrics):
        for size in size_windows:
            _, metrics, metrics_without = experiment_ampds2_data(hour, type, num_segments=11, batch_size=size, metric=metric)
            metrics['size'] = metrics.shape[0] * [size]
            metrics_without['size'] = metrics_without.shape[0] * [size]
            final_data_metrics.append(metrics)
            final_data_metrics_without.append(metrics_without)
            write_to_csv(filename='metrics', 
                    data=metrics, 
                    path = ['..', 'results', 'ampds', f'{type}', f'{size}', 'tabular'])
            write_to_csv(filename='metrics_without_deviation_and_matching', 
                    data=metrics_without, 
                    path = ['..', 'results', 'ampds', f'{type}', f'{size}', 'tabular'])
        
        final_data_metrics_df = pd.concat(final_data_metrics, ignore_index=True, sort=False)
        write_to_csv(filename='final_evalution_metrics', 
                    data=final_data_metrics_df, 
                    path = ['..', 'results', 'ampds', f'{type}'])
        
        final_data_metrics_df = pd.concat(final_data_metrics_without, ignore_index=True, sort=False)
        write_to_csv(filename='final_evalution_metrics_without_deviation_and_matching', 
                    data=final_data_metrics_df, 
                    path = ['..', 'results', 'ampds', f'{type}'])
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    # experiment1(type='original')
    # experiment1(type='original_previous')
    # experiment1(type='continuous')
    # experiment1(type='continuous_previous')

    experiment2()

    # Experiment with synthetic data
    # Experiment 3-streams with 2-dimensional data
    # final_data_metrics = []
    # for stream_num in range(3):
    #     all_metrics = []
    #     for size in size_windows:
    #         print(f"Starting size {size}")
    #         # 3-streams with 2-dimensional data
    #         _, metrics = experiment_synthetic_data(num_dimentions=2, stream_number=stream_num, num_segments=10, batch_size=size)
    #         metrics['size'] = metrics.shape[0] * [size]
    #         all_metrics.append(metrics)
    #     final_stream_metrics_df = pd.concat(all_metrics, ignore_index=True, sort=False)
    #     final_data_metrics.append(final_stream_metrics_df)
    #     write_to_csv(filename='metrics', 
    #             data=final_stream_metrics_df, 
    #             path = ['..', 'results', 'synthetic', '2-dim', 'tabular', f'stream {stream_num}', f'{size}'])
    
    # final_data_metrics_df = pd.concat(final_data_metrics, ignore_index=True, sort=False)
    # write_to_csv(filename='final_evalution_metrics', 
    #             data=final_data_metrics_df, 
    #             path = ['..', 'results', 'synthetic', '2-dim', 'tabular'])

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
