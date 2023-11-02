# All functions that will be used for preprocessing the data, 
# it will be added here.
import pandas as pd
import os
from itertools import product

from src.utils import calculate_hyper_rectangle_features

def syntethic(num_dimentions=2, num_streams=3, num_windows=10, batch_size=10):
    """
    Preprocessing the syntethic data. The data is presented in 2 or 8 dimentional data

    :param: dimentions - number of features that data has. Possible values are 2 or 8.

    :returns: pre-processed syntethic data
    """
    clustering = []
    
    # read the data 
    for stream in range(1, num_streams):
        for segment in range(num_windows):
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'syntethic', f'{num_dimentions}-dim', f'seed_75_stream_{stream}_segment_{segment}_createdelete=False.csv'))

            for index in range(0, df.shape[0], batch_size):
                new_df = df.iloc[index:index + batch_size, :]
                clustering.append({
                    'data': new_df.drop(['cluster'], axis=1),
                    'stream': stream,
                    'segment': segment,
                    'target': new_df['cluster']
                })
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


def ampds():
    """
    
    """
    pass