# All functions that will be used for preprocessing the data, 
# it will be added here.
import pandas as pd
import os
import numpy as np

from utils import calculate_hyper_rectangle_features

def s1(num_segments: int, batch_size: int, type: str):
    """
    Preprocessing the S1 data.

    :param: num_segments
    :param: batch_size

    :returns: pre-processed s1 data
    """
    clustering = []
    
    # read the data
    for segment in range(1, num_segments + 1):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 's1', f'{type}', f'{segment}.csv'))

        if batch_size:
            for index in range(0, df.shape[0], batch_size):
                new_df = df.iloc[index:index + batch_size, :]
                clustering.append({
                    'data': new_df.drop(['cluster', '_id'], axis=1),
                    'stream': -1,
                    'segment': segment,
                    'target': list(new_df['cluster'].values),
                    'is_included': [False] * new_df.shape[0],
                    'ids': list(new_df['_id'].values)
                })
        else:
            clustering.append({
                'data': df.drop(['cluster'], axis=1),
                'segment': segment,
                'target': df['cluster']
            })
    
    list_clusters_with_metrics = []
    for cluster in clustering:
        df = cluster['data']
        # find the high, low and mean vectors of each dataframe
        cluster_metrics = calculate_hyper_rectangle_features(df)
        cluster_metrics['segment'] = cluster['segment']
        cluster_metrics['stream'] = cluster['stream']
        cluster_metrics['cluster_value_count'] = batch_size
        list_clusters_with_metrics.append(cluster_metrics)

    # calculate the high, low and mean of each window    
    # save the data somewhere as files

    return {
        "clustering": clustering,
        "clustering_metrics": list_clusters_with_metrics
    }

def synthetic(num_dimentions=2, stream_number=0, num_segments=10, batch_size=10):
    """
    Preprocessing the S1 data. The data is presented in 2 or 8 dimentional data

    :param: dimentions - number of features that data has. Possible values are 2 or 8.

    :returns: pre-processed synthetic data
    """
    clustering = []
    
    # read the data
    for segment in range(1, num_segments):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', f'{num_dimentions}-dim', f'seed_75_stream_{stream_number}_segment_{segment}_createdelete=False.csv'))

        if batch_size:
            for index in range(0, df.shape[0], batch_size):
                new_df = df.iloc[index:index + batch_size, :]
                clustering.append({
                    'data': new_df.drop(['cluster'], axis=1),
                    'stream': stream_number,
                    'segment': segment,
                    'target': new_df['cluster']
                })
        else:
            clustering.append({
                'data': df.drop(['cluster'], axis=1),
                'stream': stream_number,
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
    Preprocessing the AMPDS2 data.

    :param: hour
    :param: type
    :param: num_segments
    :param: batch_size

    :returns: pre-processed AMPDS2 data
    """
    clustering = []
    
    # read the data 
    for segment in range(1, num_segments + 1):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ampds', 'segmented_data', f'{hour}H_{type}_segment_{segment}.csv'))
        df.drop(['Unnamed: 0'], axis=1, inplace=True)

        if batch_size:
            for index in range(0, df.shape[0], batch_size):
                new_df = df.iloc[index:index + batch_size, :]
                clustering.append({
                    'data': new_df,
                    'stream': -1,
                    'segment': segment,
                    'target': new_df.shape[0] * [np.nan],
                    'is_included': [False] * new_df.shape[0]
                })
        else:
            clustering.append({
                'data': df,
                'segment': segment,
                'target': new_df.shape[0] * [np.nan]
            })
    
    list_clusters_with_metrics = []
    for cluster in clustering:
        df = cluster['data']
        # find the high, low and mean vectors of each dataframe
        cluster_metrics = calculate_hyper_rectangle_features(df)
        cluster_metrics['stream'] = cluster['stream']
        cluster_metrics['segment'] = cluster['segment']
        cluster_metrics['cluster_value_count'] = batch_size
        list_clusters_with_metrics.append(cluster_metrics)

    # calculate the high, low and mean of each window    
    # save the data somewhere as files

    return {
        "clustering": clustering,
        "clustering_metrics": list_clusters_with_metrics
    }