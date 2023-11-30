# All functions that will be used for preprocessing the data, 
# it will be added here.
import pandas as pd
import os
from itertools import product

from src.utils import calculate_hyper_rectangle_features

def synthetic(num_dimentions=2, num_streams=3, num_segments=10, batch_size=100):
    """
    Preprocessing the synthetic data. The data is presented in 2 or 8 dimentional data

    :param: dimentions - number of features that data has. Possible values are 2 or 8.

    :returns: pre-processed synthetic data
    """
    clustering = []
    
    # read the data 
    for stream in range(1, num_streams):
        for segment in range(num_segments):
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', f'{num_dimentions}-dim', f'seed_75_stream_{stream}_segment_{segment}_createdelete=False.csv'))

            if batch_size:
                for index in range(0, df.shape[0], batch_size):
                    new_df = df.iloc[index:index + batch_size, :]
                    clustering.append({
                        'data': new_df.drop(['cluster'], axis=1),
                        'stream': stream,
                        'segment': segment,
                        'target': new_df['cluster']
                    })
            else:
                clustering.append({
                    'data': df.drop(['cluster'], axis=1),
                    'stream': stream,
                    'segment': segment,
                    'target': df['cluster']
                })
    
    list_clusters_with_metrics = []
    for cluster in clustering:
        df = cluster['data']
        # find the high, low and mean vectors of each dataframe
        cluster_metrics = calculate_hyper_rectangle_features(df)
        cluster_metrics['stream'] = cluster['stream']
        cluster_metrics['segment'] = cluster['segment']
        list_clusters_with_metrics.append(cluster_metrics)

    # calculate the high, low and mean of each window    
    # save the data somewhere as files

    return {
        "clustering": clustering,
        "clustering_metrics": list_clusters_with_metrics
    }


def ampds(hour: int, type: str, num_segments: int, batch_size: int):
    """
    
    """
    clustering = []
    
    # read the data 
    for segment in range(num_segments):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ampds', 'segmented_data', f'{hour}H_{type}_segment_{segment}.csv'))
        df = df.drop(['Unnamed: 0'], axis=1)
        
        if batch_size:
            for index in range(0, df.shape[0], batch_size):
                new_df = df.iloc[index:index + batch_size, :]
                clustering.append({
                    'data': new_df,
                    'stream': None,
                    'segment': segment
                })
        else:
            clustering.append({
                'data': df,
                'stream': None,
                'segment': segment
            })
    
    list_clusters_with_metrics = []
    for cluster in clustering:
        df = cluster['data']
        # find the high, low and mean vectors of each dataframe
        cluster_metrics = calculate_hyper_rectangle_features(df)
        cluster_metrics['stream'] = cluster['stream']
        cluster_metrics['segment'] = cluster['segment']
        list_clusters_with_metrics.append(cluster_metrics)

    # calculate the high, low and mean of each window    
    # save the data somewhere as files

    return {
        "clustering": clustering,
        "clustering_metrics": list_clusters_with_metrics
    }