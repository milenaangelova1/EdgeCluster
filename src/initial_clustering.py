# All functions that will be used for preprocessing the data, 
# it will be added here.
import os
from itertools import product
import pandas as pd

from src.utils import calculate_hyper_rectangle_features

def syntethic(num_dimentions=2, num_streams=0, num_segments=10):
    """
    Preprocessing the syntethic data. The data is presented in 2 or 8 dimentional data

    :param: dimentions - number of features that data has. Possible values are 2 or 8.

    :returns: pre-processed syntethic data
    """
    dfs = []
    clustering = []
    # read the data 
    for num_segment in range(num_segments):
        dfs.append(pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'syntethic', f'{num_dimentions}-dim', f'seed_75_stream_{num_streams}_segment_{num_segment}_createdelete=False.csv')))
    
    df = pd.concat(dfs)
    clustering.append({
        'data': df,
        'stream': num_streams,
        'targets': df['cluster']
    })

    list_clusters_with_metrics = []
    for cluster in clustering:
        df = cluster['data']
        cluster_labels = df['cluster'].unique()
        for label in cluster_labels:
            c = df[df['cluster'] == label]
            # find the high, low and mean vectors of each cluster
            cluster_metrics = calculate_hyper_rectangle_features(c.drop(['cluster'], axis=1))
            cluster_metrics['cluster'] = label
            cluster_metrics['stream'] = cluster['stream']
            list_clusters_with_metrics.append(cluster_metrics)
           
    # calculate the high, low and mean of each window    
    # save the data somewhere as files

    return {
        "clustering": clustering,
        "clustering_metrics": list_clusters_with_metrics
    }

def ampds():
    """
    
    """
    pass

