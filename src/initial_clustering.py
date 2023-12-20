# All functions that will be used for preprocessing the data, 
# it will be added here.
import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from clustering import kmedoids, calculate_distances, pairwise_euclidean
from metrics import calculate_connectivity, IC_av, evalutation_report
from fastdtw import fastdtw
from utils import write_to_csv

from src.utils import calculate_hyper_rectangle_features


def s1(size: int):
    """
    Preprocessing the S1 data.

    :param: size

    :returns: pre-processed s1 data
    """
    clustering = []
    # read the data 
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 's1', 'original', '0.csv'))
    df = df[df.columns[0:-1]]

    clustering.append({
        'data': df,
        'segment': [0] * df.shape[0],
        'stream': [-1] * df.shape[0],
        'targets': list(df['cluster'].values),
        'is_included': df.shape[0] * [True]
    })
    df['cluster'].value_counts().reset_index().to_csv(os.path.join(os.path.dirname(__file__), '..', 'results', 's1', 'tabular', f'{size}', f'initial_clustering_value_counts.csv'))
    df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'results', 's1', 'tabular', f'{size}', f'initial_clustering_data.csv'))

    metrics_dict = evalutation_report(data=df[df.columns[:-1]], pred_labels=df['cluster'].values)
    metrics_df =  pd.DataFrame({
            'connectivity': [metrics_dict["connectivity"]],
            'SI': [metrics_dict["SI"]],
            'S': [metrics_dict["S"]],
            "DB": [metrics_dict["DB"]]
        })
    write_to_csv(filename='initial_clustering_metrics', 
                data=metrics_df, 
                path = ['..', 'results', 's1', 'tabular', f'{size}'])

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

    clustering[0]['data']['is_included'] = clustering[0]['data'].shape[0] * [True]

    return {
        "clustering": clustering,
        "clustering_metrics": list_clusters_with_metrics
    }

def synthetic(num_dimentions=2, stream_number=0, size=3):
    """
    Preprocessing the synthetic data. The data is presented in 2 or 8 dimentional data

    :param: dimentions - number of features that data has. Possible values are 2 or 8.

    :returns: pre-processed synthetic data
    """
    clustering = []
    # read the data 
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', f'{num_dimentions}-dim', f'seed_75_stream_{stream_number}_segment_0_createdelete=False.csv'))

    clustering.append({
        'data': df,
        'stream': [stream_number] * df.shape[0],
        'segment': [0] * df.shape[0],
        'targets': list(df['cluster'].values),
        'is_included': df.shape[0] * [True]
    })
    df['cluster'].value_counts().reset_index().to_csv(os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', f'{num_dimentions}-dim', 'tabular', f'stream {stream_number}', f'{size}', f'initial_clustering_value_counts.csv'))
    df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', f'{num_dimentions}-dim', 'tabular', f'stream {stream_number}', f'{size}', f'initial_clustering_data.csv'))

    metrics_dict = evalutation_report(data=df[df.columns[:-1]], pred_labels=df['cluster'].values)
    metrics_df =  pd.DataFrame({
            'connectivity': [metrics_dict["connectivity"]],
            'SI': [metrics_dict["SI"]],
            'S': [metrics_dict["S"]],
            "DB": [metrics_dict["DB"]]
        })
    write_to_csv(filename='initial_clustering_metrics', 
                data=metrics_df, 
                path = ['..', 'results', 'synthetic', f'{num_dimentions}-dim', 'tabular', f'stream {stream_number}', f'{size}'])

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

    clustering[0]['data']['is_included'] = clustering[0]['data'].shape[0] * [True]

    return {
        "clustering": clustering,
        "clustering_metrics": list_clusters_with_metrics
    }

def initial_kmedoids_clustering(data, mini, maxi, iterations, distances=None, centroids=None):
    if distances is None:
        distances = calculate_distances(data.to_numpy(copy=True))

    silhouette_scores, silhouette_clusters = [],[]
    ic_scores, ic_clusters = [],[]
    connectivity_scores, connectivity_clusters = [],[]
    dfs = []
    for num_clusters in range(mini, maxi):
        best_silhouette, best_ic, best_connectivity = None, None, None
        sil_cluster, ic_cluster, conn_cluster = None, None, None
        
        print('num clusters: ',num_clusters)
        j = 0
        while j < iterations:
            j += 1
            try:
                C, M = kmedoids(distances, num_clusters)
            except Exception as e:
                continue
            clusters = [None for x in range(len(data))]
            df = pd.DataFrame(data)
            for i in range(len(M)):
                for index in C[i]:
                    clusters[index] = M[i]
            df['cluster'] = clusters
            try:
                silhouette = silhouette_score(distances, clusters, metric='precomputed')
                if best_silhouette is None or silhouette > best_silhouette:
                    best_silhouette = silhouette
                    sil_cluster = clusters

                connectivity = calculate_connectivity(data, 
                                                    clusters,
                                                    [x for x in range(data.shape[1])],
                                                    10, distance_matrix=distances)['CONN'].sum()
                if best_connectivity is None or connectivity < best_connectivity:
                    best_connectivity = connectivity
                    conn_cluster = clusters

                ic = IC_av(distances, clusters)[0]
                if best_ic is None or ic < best_ic:
                    best_ic = ic
                    ic_clusters = clusters
            except:
                continue

        silhouette_scores.append(best_silhouette)
        silhouette_clusters.append(sil_cluster)
        ic_scores.append(best_ic)
        ic_clusters.append(ic_cluster)
        connectivity_scores.append(best_connectivity)
        connectivity_clusters.append(conn_cluster)
        dfs.append(df)

    return [silhouette_scores,
            silhouette_clusters,
            ic_scores,
            ic_clusters,
            connectivity_scores,
            connectivity_clusters,
            ], pd.concat(dfs)

def _ampds():
    files = ['elec', 'water', 'gas', 'weather', 'all']
    distance_functions = [pairwise_euclidean, fastdtw_wrapper]
    mini = 2
    maxi = 11
    iterations = 250
    hours = [1,2,3,4,6,8]
    # hours = [8]

    for hour in hours:
        for k in range(1):
        #for k in range(12):
            for i in range(len(distance_functions)):
                for file in files:
                    print("Starting", file)
                    data = pd.read_csv(f'data/ampds/segmented_data/{hour}H_{file}_segment_{k}.csv', index_col=0, parse_dates=True)
                    #data = data.apply(zscore, axis=1, result_type='expand')
                    distances = pd.DataFrame(distance_functions[i](data.to_numpy(copy=True)))

                    returns, initial_clustering = initial_kmedoids_clustering(data.to_numpy(copy=True), mini, maxi, iterations, distances.to_numpy(copy=True))

                    index = [x for x in range(mini, maxi)]
                    cols = ["sil", "sil_clusters", "ic", "ic_clusters", "conn", "conn_clusters"]
                    df = pd.DataFrame(columns = cols)
                    for j in range(len(returns)):
                        df[cols[j]] = returns[j]
                    df.index = index
                    initial_clustering.to_csv(f'data/ampds/initial_clusterings/{hour}H_initial_clustering_{file}_{distance_functions[i].__name__}_seg_{k}.csv')
                    df.to_csv(f'data/ampds/initial_clusterings/{hour}H_initial_{file}_{distance_functions[i].__name__}_seg_{k}.csv')

                    # print("Starting", file)
                    # data = pd.read_csv(f'data/AMPds2/z_{file}_segment_{k}.csv', index_col=0, parse_dates=True)
                    # #data = data.apply(zscore, axis=1, result_type='expand')
                    # distances = pd.DataFrame(distance_functions[i](data.to_numpy(copy=True)))

                    # returns = initial_kmedoids_clustering(data.to_numpy(copy=True), mini, maxi, iterations, distances.to_numpy(copy=True))

                    # index = [x for x in range(mini, maxi)]
                    # cols = ["sil", "sil_clusters", "ic", "ic_clusters", "conn", "conn_clusters"]
                    # df = pd.DataFrame(columns = cols)
                    # for j in range(len(returns)):
                    # 	df[cols[j]] = returns[j]
                    # df.index = index
                    # df.to_csv(f'data/AMPds2/initial_clusterings/z_initial_{file}_{distance_functions[i].__name__}_seg_{k}.csv')

def fastdtw_wrapper(data):
    return np.array([[fastdtw(data[x], data[y])[0] for x in range(len(data))]for y in range(len(data))])


def ampds(hour: int, type: str):
    clustering = []
    # read the data 
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ampds', 'initial_clusterings', f'{hour}H_initial_clustering_{type}_pairwise_euclidean_seg_0.csv'))
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    
    clustering.append({
        'data': df,
        'stream': None,
        'segment': 0,
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

    return {
        "clustering": clustering,
        "clustering_metrics": list_clusters_with_metrics
    }
