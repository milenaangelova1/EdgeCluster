# All functions that will be used for preprocessing the data, 
# it will be added here.
import pandas as pd
import os
from itertools import product

from src.utils import calculate_hyper_rectangle_features

def syntethic(num_dimentions=2, num_streams=3, num_windows=10, batch_size=1000):
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
                cluster_metrics = calculate_hyper_rectangle_features(df)
                cluster_metrics['stream'] = stream
                cluster_metrics['segment'] = segment
                
            clustering.append({
                "cluster": {
                    'data': new_df.drop(['cluster'], axis=1),
                    'stream': stream,
                    'segment': segment,
                    'target': new_df['cluster']
                },
                "cluster_metrics": cluster_metrics
            })

    return clustering


def ampds():
    """
    
    """
    pass