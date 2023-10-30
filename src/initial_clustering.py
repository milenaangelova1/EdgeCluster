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
    
    clustering.append({
        'data': pd.concat(dfs),
        'stream': num_streams
    })
    
    # find the high, low and mean vectors of each dataframe
    initial_clustering_metrics = calculate_hyper_rectangle_features(clustering)

    # calculate the high, low and mean of each window    
    # save the data somewhere as files

    return {
        "clustering": clustering,
        "clustering_metrics": initial_clustering_metrics
    }

def ampds():
    """
    
    """
    pass