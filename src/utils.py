import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os

def dist(vector1: list, vector2: list) -> float:
    """
    This method will calculate the euclidean distance between two data points.
    
    :param: 
        vector1 - numerical list
        vector2 - numerical list

    :returns:
       the distance betwen two vectors (float)
    """
    return np.linalg.norm(np.array(vector1) - np.array(vector2))

def calculate_hyper_rectangle_features(cluster: list) -> dict:
    """
    This method will calculate three vectors.
     - highs: vector of all max values in the current window
     - lows: vector of all min values in the current window
     - means: a vector of all mean values that are found from the vectors above (between highs and lows)

    :param:
        clustering (dict) - dictionary that has the following information:
        {
            "data" - real clustering,
            "stream" - which stream
        }

    :returns:
        a list of clusters: each cluster is represented by three vectors: highs, lows, and means, cluster's label and which stream is.
        keys: high, low, mean, cluster_label, stream
    """
  
    highs = cluster.max()
    lows = cluster.min()
    means = (highs + lows) / 2

    return {
            "high": np.array(highs), 
            "low": np.array(lows),
            "mean": np.array(means)
    }

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
        "mean": np.array(means),
        "cluster": c['cluster'], 
        "stream": c['stream']
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
            "dist": dist(w['mean'], c['mean']),
            "cluster": c
        })
    sorted_results = sorted(results, key=lambda x: x['dist'], reverse=False)
    min_dist = sorted_results[0]['cluster']
    sorted_results.append({
        "window": w,
        "cluster": c,
        "dist": min_dist
    })
    segment = w['segment']
    stream = w['stream']
    
    # TODO: write the results in json or csv file
    # with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'syntethic', f'closed_cluster_segment_{segment}_stream_{stream}.json'), 'w') as f:
    #     dump = json.dumps(sorted_results, cls=NumpyEncoder)
    #     json.dump(dump, f)
   
    return min_dist

def update_initial_clustering(cluster_metrics, initial_clustering):
    index_cluster = None
    for index, cluster in enumerate(initial_clustering):
        if cluster['cluster'] == cluster_metrics['cluster'] and cluster['stream'] == cluster_metrics['stream']:
            index_cluster = index
            break
    if index_cluster:
        initial_clustering[index_cluster]['high'] = cluster_metrics['high']
        initial_clustering[index_cluster]['low'] = cluster_metrics['low']
        initial_clustering[index_cluster]['mean'] = cluster_metrics['mean']

def write_data_to_csv(filename, data):
    pass


def draw_graph(df: pd.DataFrame, df_metrics, filename: str, title: str, xaxis_label: str, yaxis_label: str, num_dimentions: str, color_pallete: str):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    
    data = df.values
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=color_pallete)

    highs, lows, means, clusters = preprocess_metrics(df_metrics)
    plt.scatter(highs[:, 0], highs[:, 1], c=clusters, cmap=color_pallete, marker='x', s=100)
    plt.scatter(lows[:, 0], lows[:, 1], c=clusters, cmap=color_pallete, marker='x', s=100)
    plt.scatter(means[:, 0], means[:, 1], color=['black'], marker='o', s=100)

    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title)

    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'results', 'syntethic', f'{num_dimentions}-dim', f'{filename}.png'))
    plt.show()

def preprocess_metrics(list_of_clusters):
    highs, lows, means, clusters = [], [], [], []
   
    for cluster in list_of_clusters:
        highs.append(pd.DataFrame({"0": [cluster['high'][0]], "1": [cluster['high'][1]]}))
        lows.append(pd.DataFrame({"0": [cluster['low'][0]], "1": [cluster['low'][1]]}))
        means.append(pd.DataFrame({"0": [cluster['mean'][0]], "1": [cluster['mean'][1]]}))
        clusters.append(pd.DataFrame({'cluster': [cluster['cluster']]}))
    
    highs = pd.concat(highs)
    lows = pd.concat(lows)
    means = pd.concat(means)
    clusters = pd.concat(clusters)
    return highs.values, lows.values, means.values, clusters.values

def compare_dicts(dict1: dict, dict2: dict):
    flag = False
    if np.array_equal(dict1['high'],dict2['high']) and np.array_equal(dict1['low'],dict2['low']) and np.array_equal(dict1['mean'], dict2['mean']) and dict1['cluster'] == dict2['cluster'] and dict1['stream'] == dict2['stream']:
        flag = True
    return flag