# All functions that will be used for preprocessing the data, 
# it will be added here.
import pandas as pd
import os
from itertools import product

from src.utils import calculate_hyper_rectangle_features

def syntethic(num_dimentions=2, num_streams=3, num_windows=10):
    """
    Preprocessing the syntethic data. The data is presented in 2 or 8 dimentional data

    :param: dimentions - number of features that data has. Possible values are 2 or 8.

    :returns: pre-processed syntethic data
    """

    windows = []
    windows_metrics = []
    clustering = []

    
    # read the data 
    for stream in range(1, num_streams):
        dfs = []
        for segment in range(num_windows):
            dfs.append(pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'syntethic', f'{num_dimentions}-dim', f'seed_75_stream_{stream}_segment_{segment}_createdelete=False.csv')))
    
        clustering.append({
            'data': pd.concat(dfs),
            'stream': stream
        })
    
    # find the high, low and mean vectors of each dataframe
    clustering_metrics = calculate_hyper_rectangle_features(clustering)

    # calculate the high, low and mean of each window    
    # save the data somewhere as files

    return {
        "clustering": clustering,
        "clustering_metrics": clustering_metrics
    }


def ampds():
    """
    
    """
    pass