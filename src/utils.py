from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
from colors import COLORS


from metrics import evalutation_report

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

def calculate_hyper_rectangle_features(cluster: pd.DataFrame) -> dict:
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
    # todo: change the mean formula
    # means = cluster.mean()
    means = (highs + lows) / 2

    return {
            "high": np.array(highs), 
            "low": np.array(lows),
            "mean": np.array(means)
    }

def recalculate_cluster_params(c: dict, w: dict) -> dict:
    """
    Recalculating the low, high value vectors of c.

    :param: c - a dict with keys: high, low, and mean.
    :param: w - a dict with keys: high, low, and mean.

    :returns: c - with new values for high, low and mean.
    """

    average_high = (c["high"] + w["high"]) / 2
    average_low = (c["low"] + w["low"]) / 2
    average_mean = (c["mean"] + w["mean"]) / 2

    return {
        "high": np.array(average_high), 
        "low": np.array(average_low),
        "mean": np.array(average_mean),
        "cluster": c["cluster"], 
        "segment": c["segment"],
        "stream": c["stream"]
    }

def recalculate_cluster_vectors(c: dict, w: dict) -> dict:
    """
    Recalculating the low, high value vectors of c.

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
        "cluster": c["cluster"], 
        "segment": c["segment"],
        "stream": c["stream"]
    }

def remove_cluster_metrics(cluster, initial_clustering):
    found_cluster = list(filter(lambda x: x['cluster']==cluster['cluster'], initial_clustering['clustering_metrics']))[0]
    index = initial_clustering['clustering_metrics'].index(found_cluster)
    del(initial_clustering['clustering_metrics'][index])
    
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
    min_dist = None
    if len(sorted_results) != 0:
        min_dist = sorted_results[0]['cluster']
        sorted_results.append({
            "window": w,
            "cluster": min_dist['cluster'],
            "dist": min_dist
        })
    
    return min_dist

def update_initial_clustering(cluster_metrics, initial_clustering):
    index_cluster = None
    for index, c in enumerate(initial_clustering):
        if c['cluster'] == cluster_metrics['cluster'] and c['stream'] == cluster_metrics['stream']:
            index_cluster = index
            break
    initial_clustering[index_cluster]['high'] = cluster_metrics['high']
    initial_clustering[index_cluster]['low'] = cluster_metrics['low']
    initial_clustering[index_cluster]['mean'] = cluster_metrics['mean']

def remove_closed_cluster_metrics(closed_cluster, initial_clustering):
    index_cluster = None
    for index, cluster in enumerate(initial_clustering):
        if cluster['cluster'] == closed_cluster['cluster'] and cluster['stream'] == closed_cluster['stream']:
            index_cluster = index
            break
    del(initial_clustering[index_cluster])

def remove_cluster_from_clustering(initial_clustering, cluster_for_removing):
    index = next((index for (index, d) in enumerate(initial_clustering) if d["cluster"] == cluster_for_removing['cluster']), None)
    del initial_clustering[index]
    return initial_clustering

def write_to_csv(filename, data, path):
    if data.empty:
        return
    path = os.path.join(os.path.join(os.path.dirname(__file__), *path))
    if not os.path.isdir(path):
        os.makedirs(path)
    data.to_csv(os.path.join(path, f'{filename}.csv'), index=False, sep=',')

def preprocessing_final_dataset(clustering: list):
    final_clustering = clustering[0]
    df = final_clustering['data']
    df['target'] = final_clustering['targets']
    df['segment'] = final_clustering['segment']
    df['stream'] = final_clustering['stream']
    if 'ids' in final_clustering.keys():
        df['ids'] = final_clustering['ids']
    return df

def _draw_graph(df: pd.DataFrame, df_metrics, window_metrics, filename: str, title: str, xaxis_label: str, yaxis_label: str, color_pallete: str, batch_size: int, path: str):
    # plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.clf()

    if len(df_metrics) == 0:
        return
    highs, lows, means, clusters = preprocess_metrics(df_metrics)
    if not df.empty:
        data = df.values
        scatter = plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap=color_pallete)
        # plt.legend(map(lambda x: 'Cluster ' + str(x), clusters))
        plt.legend(handles=scatter.legend_elements()[0], labels=map(lambda x: 'Cluster ' + str(x), clusters))

    # plt.scatter(highs[:, 0], highs[:, 1], c=clusters, cmap=color_pallete, marker='x', s=100)
    # plt.scatter(lows[:, 0], lows[:, 1], c=clusters, cmap=color_pallete, marker='x', s=100)
    plt.scatter(means[:, 0], means[:, 1], color=['black'], marker='o', s=100)

    if window_metrics:
        point_1, point_2, point_3, point_4 = calculate_points(highs, lows)
        x_values = [[point_1[0], point_4[0]], [point_4[0], point_2[0]], [point_2[0], point_3[0]], [point_3[0], point_1[0]]]
        y_values = [[point_1[1], point_4[1]], [point_4[1], point_2[1]], [point_2[1], point_3[1]], [point_3[1], point_1[1]]]
        plt.plot(x_values, y_values,'r--')

        highs, lows, means, clusters = preprocess_metrics(window_metrics)
        # plt.scatter(highs[:, 0], highs[:, 1], color=['green'], marker='s', s=100)
        # plt.scatter(lows[:, 0], lows[:, 1], color=['black'], marker='s', s=100)
        # plt.scatter(means[:, 0], means[:, 1], color=['black'], marker='o', s=100)

        point_1, point_2, point_3, point_4 = calculate_points(highs, lows)
        x_values = [[point_1[0], point_4[0]], [point_4[0], point_2[0]], [point_2[0], point_3[0]], [point_3[0], point_1[0]]]
        y_values = [[point_1[1], point_4[1]], [point_4[1], point_2[1]], [point_2[1], point_3[1]], [point_3[1], point_1[1]]]
        plt.plot(x_values, y_values,'b--')
    # plt.legend()

    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.title(title)
    
    path = os.path.join(os.path.dirname(__file__), *path)
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, f'{batch_size}', f'{filename}.png'))
    plt.ioff()

def get_width(cluster):
    point_1 = cluster['low']
    point_2 = [cluster['high'][0], cluster["low"][1]]
    return math.sqrt(math.pow(point_1[0] - point_2[0], 2) + math.pow(point_1[1] - point_2[1], 2))

def get_height(cluster):
    point_3 = [cluster['low'][0], cluster['high'][1]]
    point_1 = cluster['low']
    return math.sqrt(math.pow(point_1[0] - point_3[0], 2) + math.pow(point_1[1] - point_3[1], 2))

def get_coordinates(clusters):
    colors = list(COLORS.keys())
    coordinates = []
    for cluster in clusters:
        left, width = cluster['low'][0], get_width(cluster)
        bottom, height = cluster['low'][1], get_height(cluster)
        mean = cluster['mean']

        right = left + width
        top = bottom + height

        color = colors[cluster['cluster']]
        coordinates.append({"left": left, "width": width, "bottom": bottom, "height": height, "right": right, "top": top, "mean": mean, "color": color, "cluster": cluster['cluster']})
    return coordinates

def plot_rectangles(clusters: list):
    # build a rectangle in axes coords
   
    handles = []
    coordinates = get_coordinates(clusters)
    clusters = list(map(lambda x: f'Cluster {x["cluster"]}', clusters))
    _, ax = plt.subplots()

    for _, coord in enumerate(coordinates):
        left = coord['left']
        bottom = coord['bottom']
        width = coord['width']
        height = coord['height']
        mean = coord['mean']

        ax.plot(mean[0], mean[1], 'o', color=COLORS[coord['color']])

        p = patches.Rectangle(
                (left, bottom), width, height,
                fill=True, clip_on=False,
                alpha=0.6,
                facecolor=COLORS[coord['color']]
                )

        ax.add_patch(p)
        handles.append(p)
    return handles, clusters

def compare_dicts(dict1: dict, dict2: dict):
    flag = False
    if np.array_equal(dict1['high'],dict2['high']) and np.array_equal(dict1['low'],dict2['low']) and np.array_equal(dict1['mean'], dict2['mean']) and dict1['cluster'] == dict2['cluster'] and dict1['stream'] == dict2['stream']:
        flag = True
    return flag

def calculate_points(higher_point: dict, lower_point: dict):
    # lower point
    point_1 = [lower_point[0][0], lower_point[0][1]]
    # higher point
    point_2 = [higher_point[0][0], higher_point[0][1]]

    point_3 = [point_1[0], point_2[1]]
    point_4 = [point_2[0], point_1[1]]
    return point_1, point_2, point_3, point_4

def draw_graph(df_metrics, window_metrics, filename: str, title: str, xaxis_label: str, yaxis_label: str, batch_size: int, path: str, initial_graph=False):
    plt.rcParams["figure.autolayout"] = True
    plt.clf()

    if len(df_metrics) == 0:
        return
    
    if initial_graph:
        handles, clusters = plot_rectangles(df_metrics)
    else:
        handles, clusters = plot_rectangles(df_metrics['clustering'])

    if window_metrics:
        highs, lows = preprocess_metrics([df_metrics['closed_cluster']])
        # plot cluster vectors
        point_1, point_2, point_3, point_4 = calculate_points(highs, lows)
        x_values = [[point_1[0], point_4[0]], [point_4[0], point_2[0]], [point_2[0], point_3[0]], [point_3[0], point_1[0]]]
        y_values = [[point_1[1], point_4[1]], [point_4[1], point_2[1]], [point_2[1], point_3[1]], [point_3[1], point_1[1]]]
        cluster_plot = plt.plot(x_values, y_values,'r--')

        highs, lows = preprocess_metrics(window_metrics)

        # plot window vectors
        point_1, point_2, point_3, point_4 = calculate_points(highs, lows)
        x_values = [[point_1[0], point_4[0]], [point_4[0], point_2[0]], [point_2[0], point_3[0]], [point_3[0], point_1[0]]]
        y_values = [[point_1[1], point_4[1]], [point_4[1], point_2[1]], [point_2[1], point_3[1]], [point_3[1], point_1[1]]]
        window_plot = plt.plot(x_values, y_values,'k--')

        for c_plot, w_plot in zip(cluster_plot, window_plot):
            handles.append(c_plot)
            handles.append(w_plot)
        clusters.append('Closed cluster')
        clusters.append('Current window')

    plt.legend(handles=handles, labels = clusters, loc='center left', bbox_to_anchor=(0.0, -0.3), ncol=3, borderaxespad=0)

    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    # plt.title(title)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    path = os.path.join(os.path.dirname(__file__), *path)
    if not os.path.isdir(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, f'{batch_size}', f'{filename}.png'))
    plt.ioff()
    
def preprocess_metrics(list_of_clusters):
    highs, lows, means, clusters = [], [], [], []
    
    if not list_of_clusters:
        return
    
    for cluster in list_of_clusters:
        highs.append(pd.DataFrame({"0": [cluster['high'][0]], "1": [cluster['high'][1]]}))
        lows.append(pd.DataFrame({"0": [cluster['low'][0]], "1": [cluster['low'][1]]}))
        means.append(pd.DataFrame({"0": [cluster['mean'][0]], "1": [cluster['mean'][1]]}))
        if 'cluster' in cluster.keys():
            clusters.append(pd.DataFrame({'cluster': [cluster['cluster']]}))
    
    highs = pd.concat(highs)
    lows = pd.concat(lows)
    means = pd.concat(means)
       
    return highs.values, lows.values


def get_label(clustering):
    label = ''
    if clustering['changes']['deviated']:
        label += 'deviated'
    elif clustering['changes']['merges']:
        label += 'merges'
    elif clustering['changes']['matched']:
        label += 'matched'
    elif clustering['changes']['windows']:
        label += 'window'
    elif clustering['changes']['merge_cluster_with_window']:
        label +='merge_cluster_with_window'
    elif clustering['changes']['merge_window_with_cluster']:
        label +='merge_window_with_cluster'
    return label

def summary(final_clustering):
    cluster_df = pd.json_normalize(final_clustering['clustering'], meta=[['high', 'low', 'mean', 'cluster', 'segment', 'stream']])
    # if 'segment' in cluster_df.columns:
    #     cluster_df = cluster_df.drop(['segment', 'size'], axis=1)
    cluster_df.columns = ['cluster high', 'cluster low', 'cluster mean', 'cluster label', 'cluster segment', 'cluster stream']
    
    closed_cluster_df = pd.json_normalize(final_clustering['closed_cluster'], meta=[['high', 'low', 'mean', 'cluster', 'segment', 'stream']])
    # if 'segment' in closed_cluster_df.columns:
    #     closed_cluster_df = closed_cluster_df.drop(['segment'], axis=1)
    closed_cluster_df.columns = ['closed cluster high', 'closed cluster low', 'closed cluster mean', 'closed cluster label', 'closed cluster segment', 'closed cluster stream']

    window_df = pd.json_normalize(final_clustering['window'], meta=[['high', 'low', 'mean', 'segment', 'stream']])
    if 'cluster' in window_df.columns:
        window_df = window_df.drop(['cluster'], axis=1)
    window_df.columns = ['window high', 'window low', 'window mean', 'window segment', 'window stream']

    changes_df = pd.json_normalize(final_clustering['changes'], meta=[['deviated', 'matched', 'merges', 'windows', 'merge_cluster_with_window', 'merge_with_window']])
    changes_df.columns = ['is the cluster and the window deviated', 'is the cluster and the window matched', 'cluster labels that are merged together into a cluster', 'new clusters (labels)', 'merge a cluster with a window', 'merge a window with a cluster']

    df = pd.concat([closed_cluster_df, cluster_df, window_df, changes_df], axis=1)
    return df

def move_data(initial_clustering, clustering, list_of_windows, window_index):
    """
    :param: initial_clustering - this is the original clustering at the begining
    :param: clustering - this is the current clustering for a specific window
    :param: list of windows - all windows from the begining.
    """
    merges = clustering['changes']['merges']
    windows = clustering['changes']['windows']
    deviated = clustering['changes']['deviated']
    matched = clustering['changes']['matched']
    merge_window_with_cluster = clustering['changes']['merge_window_with_cluster']
    merge_cluster_with_window = clustering['changes']['merge_cluster_with_window']

    cluster = clustering['closed_cluster']['cluster']
    window = clustering['window']
    if merges:
        for cluster_label in merges:
            initial_clustering['clustering'][0]['data']['cluster'].replace(cluster_label, cluster, inplace=True)
        window_data, segment, stream, target, ids = find_window(list_of_windows, window_index)
        add_cluster_label(window_data, cluster)
        if not window_data is None:
            add_window_data(initial_clustering, window_data, segment, stream, target, ids)
            add_correct_clustering_labels(initial_clustering, cluster)
    elif windows:
        window_data, segment, stream, target, ids = find_window(list_of_windows, window_index)
        cluster_label = window['cluster']
        add_cluster_label(window_data, cluster_label)
        add_window_data(initial_clustering, window_data, segment, stream, target, ids)
    elif deviated:
        window_data, segment, stream, target, ids = find_window(list_of_windows, window_index)
        window_data['cluster'] = -1
        add_window_data(initial_clustering, window_data, segment, stream, target, ids, is_included=False)
    elif matched:
        window_data, segment, stream, target, ids = find_window(list_of_windows, window_index)
        window_data['cluster'] = cluster
        add_window_data(initial_clustering, window_data, segment, stream, target, ids)
    elif merge_cluster_with_window:
        window_data, segment, stream, target, ids = find_window(list_of_windows, window_index)
        window_data['cluster'] = cluster
        # Union between them
        add_window_data(initial_clustering, window_data, segment, stream, target, ids, is_included=True)
    elif merge_window_with_cluster:
        window_data, segment, stream, target, ids = find_window(list_of_windows, window_index)
        window_data['cluster'] = cluster
        # Union between them
        add_window_data(initial_clustering, window_data, segment, stream, target, ids, is_included=True)
        add_correct_clustering_labels(initial_clustering, cluster)

def find_indexes(df, high_vector, low_vector):
    indexes = [] 
    for index, row in df.iterrows():
        if all((np.array(list(row)) >= np.array(low_vector))) and all((np.array(list(row)) <= np.array(high_vector))):
            indexes.append(index)
    indexes = list(df.loc[df.index.difference(indexes)].index)
    return indexes

def add_correct_clustering_labels(initial_clustering, cluster):
    cluster_params = list(filter(lambda x: x['cluster']==cluster, initial_clustering['clustering_metrics']))[0]
    high_vector = list(cluster_params['high'])
    low_vector = list(cluster_params['low'])
    df = initial_clustering['clustering'][0]['data'].copy()
    df = df[df['cluster']==cluster]
    df = df[df.columns[:-2]]

    indexes = find_indexes(df, high_vector, low_vector)
    if len(indexes) > 0:
        initial_clustering['clustering'][0]['data'].loc[indexes, 'cluster'] = initial_clustering['clustering'][0]['data'].loc[indexes, 'cluster'].replace(cluster, -2)
        initial_clustering['clustering'][0]['data'].loc[indexes, 'is_included'] = initial_clustering['clustering'][0]['data'].loc[indexes, 'is_included'].replace(True, False)

def add_window_data(initial_clustering, window_data, segment, stream, target, ids, is_included=True):
    window_data['is_included'] = window_data.shape[0] * [is_included]
    df = pd.concat([initial_clustering['clustering'][0]['data'], window_data], ignore_index=True, sort=False)

    initial_clustering['clustering'][0]['data'] = df
    initial_clustering['clustering'][0]['segment'] = list(initial_clustering['clustering'][0]['segment']) + [segment] * window_data.shape[0]
    initial_clustering['clustering'][0]['stream'] = list(initial_clustering['clustering'][0]['stream']) + [stream] * window_data.shape[0]
    initial_clustering['clustering'][0]['targets'] = list(initial_clustering['clustering'][0]['targets']) + target
    if 'ids' in initial_clustering['clustering'][0]:
        initial_clustering['clustering'][0]['ids'] = list(initial_clustering['clustering'][0]['ids']) + ids
    initial_clustering['clustering'][0]['is_included'] = list(initial_clustering['clustering'][0]['is_included']) + list(window_data['is_included'].values)
            
def find_window(list_of_windows, window_index):
    window_data = list_of_windows['clustering'][window_index]
    ids = None
    if 'ids' in window_data.keys():
        ids = window_data['ids']
    return window_data['data'], window_data['segment'], window_data['stream'], window_data['target'], ids

def add_cluster_label(clustering, cluster):
    clustering["cluster"] = clustering.shape[0] * [cluster]

def evaluation_metrics(final_df, segment, is_included=False, true_labels= True, metric='euclidean'):
    df = final_df.copy()
    ids = None
    if is_included:
        df = df[df["is_included"] == True]

    if df.empty:
        return pd.DataFrame({
        # 'connectivity': [metrics_dict["connectivity"]],
        "F1": [np.nan],
        "SI": [np.nan],
        "JI": [np.nan],
        # "homogeneity": [np.nan],
        # "RI": [np.nan],
        # "ARI": [np.nan],
        # "MI": [np.nan],
        # "NMI": [np.nan],
        # "AMI": [np.nan],
        # "CS": [np.nan],
        # "V": [np.nan],
        # "FMI": [np.nan],
        # "S": [np.nan],
        # "DB": [np.nan],
        "IC_av": [np.nan],
        "segment": [segment],
        "stream": "-"
    })

    if 'ids' in df.columns:
        ids = df['ids']
        df.drop(['ids'], axis=1, inplace=True)

    if true_labels:
        metrics_dict = evalutation_report(data=df[df.columns[:-5]], pred_labels=df["cluster"].values, true_labels=df["target"].values, ids=ids)
    else:
        metrics_dict = evalutation_report(data=df[df.columns[:-5]], pred_labels=df["cluster"].values, ids=ids, metric=metric)
    metrics_df =  pd.DataFrame({
        # 'connectivity': [metrics_dict["connectivity"]],
        "F1": [metrics_dict["F1"]],
        "SI": [metrics_dict["SI"]],
        "JI": [metrics_dict["JI"]],
        # "homogeneity": [metrics_dict["homogeneity"]],
        # "RI": [metrics_dict["RI"]],
        # "ARI": [metrics_dict["ARI"]],
        # "MI": [metrics_dict["MI"]],
        # "NMI": [metrics_dict["NMI"]],
        # "AMI": [metrics_dict["AMI"]],
        # "CS": [metrics_dict["CS"]],
        # "V": [metrics_dict["V"]],
        # "FMI": [metrics_dict["FMI"]],
        # "S": [metrics_dict["S"]],
        # "DB": [metrics_dict["DB"]],
        "IC_av": [metrics_dict["IC_av"]],
        "cluster count": len(df["cluster"].unique()), 
        "segment": [segment],
        "stream": "-"
    })
    return metrics_df

def update_segments_dict(segments:dict, window_segement: int, initial_clustering: dict):
    segments[window_segement].append(initial_clustering)

def metrics_by_segments(segments, batch_size, path, true_labels, type='continuous', metric='euclidean'):
    metrics = []
    metrics_without = []
    data = pd.DataFrame()
    for segment in segments.keys():
        if segment == 0:
            data = segments[segment][-1]['clustering'][0]['data'][segments[segment][-1]['clustering'][0]['data']['segment']==0]
            seg = f'{segment}'
        else:
            length = len(segments[segment]) - 1
            if type == 'original_previous' or type == 'continuous_previous':
                data = segments[segment][length]['clustering'][0]['data'][(segments[segment][length]['clustering'][0]['data']['segment']==segment-1) | (segments[segment][length]['clustering'][0]['data']['segment']==segment)]
                seg = f'{segment-1}-{segment}'
            elif type == 'original' or type == 'continuous':
                data = segments[segment][length]['clustering'][0]['data'][segments[segment][length]['clustering'][0]['data']['segment']==segment]
                seg = f'{segment}'
        write_to_csv(filename=f'final_clustering_data_segment_{seg}', 
                    data=data,
                    path = path)
        metrics_df = evaluation_metrics(data, segment, true_labels=true_labels, metric=metric)
        metrics.append(metrics_df)

        metrics_without.append(evaluation_metrics(data, segment, is_included=True, true_labels=true_labels, metric=metric))
    return metrics, metrics_without
            