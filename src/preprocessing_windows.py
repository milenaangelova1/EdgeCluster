# All functions that will be used for preprocessing the data, 
# it will be added here.
from pandas import DataFrame, read_csv
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

    stream_segment_combinations = product(list(range(1,num_streams)), list(range(num_windows)))
    
    for num_stream, num_segment in stream_segment_combinations:
        df = read_csv(os.path.join(os.path.dirname(__file__), f'../data/syntethic/{num_dimentions}/seed_75_stream_{num_stream}_segment_{num_segment}_createdelete=False.csv'))
        windows.append({
            'data': df,
            'segment': num_segment,
            'stream': num_stream})

    for window in windows:
        windows_metrics.append(calculate_hyper_rectangle_features(window))

    # calculate the high, low and mean of each window    
    # save the data somewhere as files
    return {
        "windows": windows,
        "initial_clustering_metrics": windows_metrics
    }

def ampds():
    """
    
    """
    pass