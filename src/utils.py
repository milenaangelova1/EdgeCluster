import pandas as pd
import numpy as np
from pprint import pprint

def dist(vector1: list, vector2: list) -> float:
    """
    This method will calculate the euclidean distance between two data points.
    
    :param: 
        vector1 - numerical list
        vector2 - numerical list

    :returns:
       the distance betwen two vectors (float)
    """
    if vector1.size == 0 or vector2 == 0:
        raise ValueError("All or one of the vectors is empty!")

    return np.linalg.norm(np.array(vector1) - np.array(vector2))

def __calculate_hyper_rectangle_features(df: pd.DataFrame) -> dict:
    """
    This method will calculate three vectors.
     - highs: vector of all max values in the current window
     - lows: vector of all min values in the current window
     - means: a vector of all mean values that are found from the vectors above (between highs and lows)

    :param:
        window - a dataframe that contains the current data (window)

    :returns:
        a dict with three vectors inside: highs, lows, and means
        keys: high, low, mean
        values: vectors (ndarrays)
    """
    if df.empty:
        raise ValueError("The dataset is empty!")
    
    highs = df.max()
    lows = df.min()
    means = (highs + lows) / 2

    return {
        "high": np.array(highs), 
        "low": np.array(lows),
        "mean": np.array(means)
    }

def get_data_windows(df:pd.DataFrame):
    """
    Data windows represented like a list of windows.
    Each window will be a list of three vectors: high, low, and mean.

    :param: df - data that needs to be presented as a window
    """
    if df.empty:
        raise ValueError("The data frame is empty!")
    return __calculate_hyper_rectangle_features(df)


def get_initial_clustering(df: pd.DataFrame):
    """
    A list of clusters
    """
    pass

def merge_clusters():
    pass

def recalculate_window_params(c: dict, w: dict) -> dict:
    """
    Recalculating the window low, high value vectors of c.

    :param: c - a dict with keys: high, low, and mean.
    :param: w - a dict with keys: high, low, and mean.

    :returns: c - with new values for high, low and mean.
    """
    
    lows_df = pd.DataFrame([c['low'], w['low']])
    highs_df = pd.DataFrame([c['high'], w['high']])

    highs = highs_df.max()
    lows = lows_df.min()
    means = (highs + lows) / 2

    return {
        "high": np.array(highs), 
        "low": np.array(lows),
        "mean": np.array(means)
    }

def find_the_closest_cluster(w: dict, C: list) -> dict:
    """
    Find the closest cluster to the window w.

    :param: w - a data window
    :param: C - list of Clusters

    :return: the closest cluster
    """
    results = []
    for c in C:
        results.append({
            "dist": dist(w, c),
            "cluster": c
        })
    return pprint(sorted(results, key=lambda x: x['dist'], reverse=False))['cluster']
    
