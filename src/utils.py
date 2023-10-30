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

def calculate_hyper_rectangle_features(cluster: dict) -> dict:
    """
    This method will calculate three vectors.
     - highs: vector of all max values in the current window
     - lows: vector of all min values in the current window
     - means: a vector of all mean values that are found from the vectors above (between highs and lows)

    :param:
        cluster (dict) - dictionary that has the following information:
        {
            "data" - real clustering,
            "segment" - which data segment,
            "stream" - which stream
        }

    :returns:
        a dict with three vectors: highs, lows, and means, and returns information about which stream and segment is.
        keys: high, low, mean, stream, segment
        values: vectors (ndarrays)
    """
    if cluster.data.empty:
        raise ValueError("The dataset is empty!")
    
    highs = cluster.data.max()
    lows = cluster.data.min()
    means = (highs + lows) / 2

    return {
        "high": np.array(highs), 
        "low": np.array(lows),
        "mean": np.array(means),
        "stream": cluster['stream'],
        "segment": cluster['segment']
    }

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
    for c, stream, segment in C['data'], C['stream'], C['segment']:
        results.append({
            "dist": dist(w, c),
            "window": w,
            "stream": stream,
            "segment": segment
        })
    return pprint(sorted(results, key=lambda x: x['dist'], reverse=False))['cluster']

def write_data_to_csv(filename, data):
    pass
    
