
from scipy.sparse.csgraph import minimum_spanning_tree
import pandas as pd
import numpy as np
import copy
from clustering import calculate_distances
from sklearn.metrics import (pairwise_distances, homogeneity_score, silhouette_score, rand_score, adjusted_rand_score, fowlkes_mallows_score, completeness_score,
    v_measure_score,mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    rand_score, adjusted_rand_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    jaccard_score
    )

from scipy.spatial import distance
from scipy.stats import median_abs_deviation
from copy import deepcopy

def find_centroids(data, labels):
    _data = deepcopy(data)
    _data['cluster'] = labels
    centroids = _data.groupby('cluster').mean()
    return centroids

def evalutation_report(data, pred_labels, true_labels=[], metric='euclidean', ids=None):
    F1 = None
    _homogeneity_score = None
    SI = None
    connectivity = None
    RI = None
    ARI = None
    MI = None
    NMI = None
    AMI = None
    CS = None
    V = None
    FMI = None
    s = None
    DB = None
    JI = None
    ICav = None
    TS = None
    # distances = calculate_distances(data.to_numpy(copy=True), metric)
    # connectivity = calculate_connectivity(data, 
    #                                     pred_labels,
    #                                     [x for x in range(data.shape[1])],
    #                                     5, distance_matrix=distances)['CONN'].sum()
    if len(set(pred_labels)) > 1:
        # SI = calculate_silhouette(distances, pred_labels)
        SI = silhouette_score(data, pred_labels, n_jobs=-1)
        s = calinski_harabasz_score(data, pred_labels)
        DB = davies_bouldin_score(data, pred_labels)
    # ICav = IC_av(distances, pred_labels)

    _,coeff, TS = tempsil(ids,data.values,true_labels,s=100,kn=1000,c=1)
    centroids = find_centroids(data, pred_labels)
    ssq = calculate_ssq(data_points=data, centroids=centroids, labels=pred_labels)     

    if len(true_labels) > 1:
        _homogeneity_score = homogeneity_score(true_labels, pred_labels)
        RI = rand_score(true_labels, pred_labels)
        ARI = adjusted_rand_score(true_labels, pred_labels)
        # MI = mutual_info_score(true_labels, pred_labels)
        # NMI = normalized_mutual_info_score(true_labels, pred_labels)
        AMI = adjusted_mutual_info_score(true_labels, pred_labels)
        # CS = completeness_score(true_labels, pred_labels)
        # V = v_measure_score(true_labels, pred_labels, beta=1.0)
        # FMI = fowlkes_mallows_score(true_labels, pred_labels)

    if ids is not None:
        data['cluster'] = pred_labels
        data['_id'] = ids
        data['actual_cluster'] = true_labels
        F1 = cluster_wise_f_measure(data)
        JI = cluster_wise_jaccard(data)
    
    return {
        "connectivity": connectivity, 
        "F1": F1, 
        "SI": SI, 
        "homogeneity":_homogeneity_score,
        "RI": RI,
        "ARI": ARI,
        # "MI": MI,
        # "NMI": NMI,
        "AMI": AMI,
        # "CS": CS,
        # "V": V,
        # "FMI": FMI,
        "s": s,
        "DB": DB,
        "JI": JI,
        # "IC_av": ICav,
        "TSI": TS,
        "ssq": ssq
    }

# F1
def cluster_wise_f_measure(data):
    sum = 0
    unique_clusters_predicted = sorted(data['cluster'].unique())
    for cluster in unique_clusters_predicted:
    	predicted_cluster = data[data['cluster'] == cluster]
    	true_cluster = cluster_with_max_shared(predicted_cluster, data)
    	sum += f_measure(set(predicted_cluster['_id']), set(true_cluster['_id']))
    
    length = len(unique_clusters_predicted)
    return (sum / length)

def f_measure(pred, true):
    value = (2 * len(pred & true)) / (len(true) + len(pred))
    return value

# SI
def calculate_silhouette(distances, clusters):
    return silhouette_score(distances, clusters, metric='precomputed')

# IC_av
def IC_av(distance_matrix = None, labels = None):
    """Calculates Intra Cluster distance average of a clustering solution.

    Args:
        distance_matrix: 2-dimensional (n * n) matrix of Pandas Dataframe type containing distances between all nodes.
        labels: Clustering labels to assign each node.
    Returns:
        List containing total IC_av distance and each individual IC_av distance for each cluster.
    Raises:
        TypeError: if distance_matrix is not a Pandas.Dataframe.
        ValueError: if distance_matrix or labels is of None value.
    """
    # if distance_matrix is None:
    #     raise ValueError("distance_matrix cannot be None value")
    # if not isinstance(distance_matrix, pd.DataFrame):
    #     raise TypeError("distance_matrix must be DataFrame")
    # if labels is None:
    #     raise ValueError("labels cannot be None value")


    # MST of the distance matrix
    mst = minimum_spanning_tree(distance_matrix)

    MED_matrix = pd.DataFrame(fast_IC_av(mst))
    
    IC_score = 0
    IC_scores = [0]
    clusters = list(sorted(set(labels))) # Get clusterlabels from labels

    MED_matrix.reset_index
    MED_matrix['clusters'] = labels


    # Traverse the mst for the largest edge and calculate the IC_av score
    for cluster in clusters:
        nodes = MED_matrix.loc[MED_matrix['clusters'] == cluster].index.tolist()
        total = 0
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                total += MED_matrix[i][j]
                #total += MED_matrix[i][j]**2

        total /= len(nodes)
        IC_score += total
        IC_scores.append(total)

    # Returns total IC_av score and individual IC_av scores for each cluster
    IC_scores[0] = IC_score
    return IC_scores

def d_med(tree, i, j):
    # traverse from i to j and return the max edge value
    return traverse(tree, i, i, j)

def traverse(tree, current, parent, target):
    if current == target:   # base case
        return tree[current][parent]

    # Get indexes that connect this node to the others except the parent
    children = [x for x in range(len(tree[current])) if tree[current][x] != 0 and x != parent]

    for child in children:
        dist = traverse(tree, child, current, target)
        if dist > 0:
            if dist > tree[current][parent]:
                return dist
            return tree[current][parent]

    return -1

def fast_IC_av(mst):
    """Returns the MED distance matrix for all nodes in the graph"""

    N = mst.shape[0]

    global graph
    graph = mst.toarray()
    
    # Preparation
    # for a non-directed graph, we need to symmetrize the distances
    # this is necessary since the minimum spanning tree only delivers edges in one direction
    for i in range(N):
        for j in range(i + 1, N):
            if graph[j, i] >= graph[i, j]:
                graph[i, j] = graph[j, i]
            else:
                graph[j, i] = graph[i, j]


    # 1) find all leaf nodes
    global leaf_list
    leaf_list = find_leaves(graph)
    # print("Leaves", leaf_list)

    # Set all distances with no direct connection to infinity
    graph[np.where(graph == 0)] = np.nan

    # Graph[i,i] should be zero
    graph.flat[::N + 1] = 0
    
    # 2) Store the results
    # MED_matrix saves the MED from node a to b 
    # MED is positive and symmetrical
    # Add takes the initial distances from graph
    global MED_matrix
    MED_matrix = copy.deepcopy(graph)
    

    # 3) Create a path dictionary to store started paths
    global path_list
    path_list = {}
    
    # Follow the paths while there are still leaf nodes
    while(leaf_list):
        first_node = leaf_list.pop()
        current_path = [first_node]

        neighbor_list = find_neighbors(graph, first_node, current_path)

        # since that is a leaf there should be only one neighbor
        assert len(neighbor_list) == 1

        # Calculate the distance to the next node
        dist = graph[first_node, neighbor_list[0]]

        follow_path(next_node=neighbor_list[0], current_path=current_path, dist=dist)

    return MED_matrix


def follow_path(next_node, current_path, dist):

    # Update the current path with the distance to the next node
    update_distances_node_centered(dist=dist, visited_list=current_path, shape=graph.shape[0], next_node=next_node)

    # Add the next element to the current path
    current_path.append(next_node)

    # Find path to the next elements
    neighbor_list = find_neighbors(graph, next_node, current_path)

    # Check whether an intersection or a leaf
    if not neighbor_list:
        # Remove the leaf from the leaf-list
        # Since it doesn't have to be a leaf it could be also stuck somewhere between known parts
        if next_node in leaf_list:
            leaf_list.remove(next_node)
        return

    elif len(neighbor_list) == 1:
        dist = graph[next_node, neighbor_list[0]]
        follow_path(next_node=neighbor_list[0], current_path=current_path, dist=dist)
    else:
        # Intersection case
        # Check if the current node is already in the dictionary
        if path_list.get(next_node):
            # Print already stored
            # Join lists and follow the path
            current_path = current_path + path_list.get(next_node)

            # Find the next way which is not in the joined list
            dest_node = [x for x in neighbor_list if x not in current_path]

            if not dest_node:
                # No more paths so the algorithm is finished
                print("No way to go. Empty join result")
                return
            elif len(dest_node) > 1:
                # Store the current path in the dictionary
                path_list.update({next_node:current_path})
                # Start again at the next leaf
                return

            else:
                # Follow the path with the now extended list
                # With the only left node to go
                dist = graph[next_node, dest_node[0]]
                follow_path(next_node=dest_node[0], current_path=current_path, dist=dist)

        else:
            # Store the current path in the dictionary
            path_list.update({next_node:current_path})
            # Start again at the next leaf
            return


def update_distances(dist, visited_list, shape):
    for node in range(shape):
        # Do not update distances in the already visited paths
        if node in visited_list:
            continue

        # Go through all the visited nodes and compare the distances, 
        # in case it is bigger perform an update
        for vis_node in visited_list:
            MED_matrix[vis_node, node] = np.nanmax([MED_matrix[vis_node, node], dist])
            MED_matrix[node, vis_node] = MED_matrix[vis_node, node]

def update_distances_node_centered(dist, visited_list, shape, next_node):
    # Going through the already visited nodes
    for vis_node in visited_list:
        # In case we already have a bigger distance to the new added node
        # we can skip this node since nothing will change
        # print(MED_matrix[vis_node, next_node], dist)
        if MED_matrix[vis_node, next_node] <= dist or np.isnan(MED_matrix[vis_node, next_node]):
            # Go through all nodes and update the distance
            for node in range(shape):
                # Do not update distances in the already visited paths
                if node in visited_list:
                    continue
                MED_matrix[vis_node, node] = np.nanmax([MED_matrix[vis_node, node], dist])
                MED_matrix[node, vis_node] = MED_matrix[vis_node, node]
        #else:
        #    print(next_node, visited_list, dist, vis_node)
        #    print(MED_matrix)

        # Go through all the visited nodes and compare the distances, 
        # in case it is bigger perform an update


def find_neighbors(graph, node, current_path):
    neighbor_list = []
    for ii in range(graph.shape[0]):
        if not np.isnan(graph[node, ii]) and ii not in current_path:
            neighbor_list.append(ii)

    return neighbor_list
    

def find_leaves(graph):
    """Returns a list of all leaves"""
    
    leaf_list = []

    for row in range(graph.shape[0]):
        # print(np.count_nonzero(graph[row]))
        # one non-zero element means that there is only one connection to another node
        # making it to a leaf
        if np.count_nonzero(graph[row]) == 1:
            leaf_list.append(row)

        # Iris data set 54 leaves, so more than one third
    return leaf_list

# Connectivity
def find_nearest_neighbors(sample, neighbors, n_neighbors):
    """Finds the nearest neighbors to the given sample"""
    neighbors = np.delete(neighbors, sample)
    nearest_neibour_indexes = np.argsort(neighbors)[:n_neighbors]
    return nearest_neibour_indexes

def get_connectivity(dataframe, i, j, j_index):
    """Gets the connectivity value from a given dataframe"""
    i_class = dataframe.iloc[i]["cluster"]
    j_class = dataframe.iloc[j]["cluster"]
    state = (0 if i_class == j_class else float(1)/(j_index + 1))
    return state

def connectivity_samples(X, y, n_neighbors, metric=None, distance_matrix=None, **kwds):
    """Calculates the connectivity for each sample in the dataset."""
    dataframe = pd.DataFrame(data=X, index=range(len(X)))
    dataframe["cluster"] = y
    N = len(X)
    
    if distance_matrix is None:
        distance_matrix = pairwise_distances(X, metric=metric, **kwds)
    
    connectivity_list = []
    for i in range(N):
        connectivity_sum = 0
        nearest_neighbors = find_nearest_neighbors(i, distance_matrix[i], n_neighbors)
        for j in range(n_neighbors):
            connectivity_sum += get_connectivity(dataframe, i, nearest_neighbors[j], j)
        connectivity_list.append(connectivity_sum)
    return connectivity_list

def calculate_connectivity(X_train, y_train, columns, n_neighbors, metric=None, distance_matrix=None):
    si_samples = connectivity_samples(X_train, y_train, n_neighbors, metric, distance_matrix)
    dataframe = pd.DataFrame(data=X_train, columns=columns, index=range(0, len(y_train)))
    dataframe["CONN"] = si_samples
    dataframe["cluster"] = y_train
    #sorted_dataframe = dataframe.sort_values(["CONN"], ascending=[True])
    return dataframe


# Jaccard index
def cluster_with_max_shared(predicted_cluster, data):
	predicted = set(predicted_cluster['_id'])
	true = data['actual_cluster'].unique()
	max_cluster = 0
	max_session = 0

	for label in true:
		cluster = data[data['actual_cluster'] == label]
		same_cluster = len(predicted & set(cluster['_id']))
		if same_cluster > max_cluster:
			max_cluster = same_cluster
			max_session = label
	return data[data['actual_cluster'] == max_session]

def cluster_wise_jaccard(data):
	sum = 0
	unique_clusters_predicted = sorted(data['cluster'].unique())
	for cluster in unique_clusters_predicted:
		predicted_cluster = data[data['cluster'] == cluster]
		true_cluster = cluster_with_max_shared(predicted_cluster, data)
		sum += jaccard_measure(set(predicted_cluster['_id']), set(true_cluster['_id']))

	length = len(unique_clusters_predicted)
	return (sum / length)


def jaccard_measure(y_true, y_pred):
    return (len(y_pred & y_true)) / (len(y_true) + len(y_pred) - len(y_true & y_pred))

def temp_centroid_coherence(X):
    A,B = X[1:,:],X[:-1,:]   
    d = A-B
    d = np.linalg.norm(d, axis=1)
    if len(d):
        return mad(d,False)
    else:
        return 0

def dist2oc(x,M,l):
    d = distance.cdist(M,x)
    clus = np.unique(l)
    meandclus = np.inf * np.ones(len(clus))
    for i,c in enumerate(clus):
        meandclus[i] = np.mean(d[l==c])
    return np.min(meandclus)

def find_knearest(x, v, k, option='nearest'):
    x=x.flatten()
    if option=='random':
        i = np.random.permutation(len(x))
    else:
        i = np.argsort((np.abs(x - v)))
    ind = i[:k]
    return x[ind]

def mad(x, x_is_int=True):
    madx = median_abs_deviation(x, scale = "normal")
    if x_is_int:
        madx = 1 if madx<1 else madx 
    else:
        madx = 1 if madx==0 else madx 
    nmads = np.abs(x-np.median(x))/madx
    outs = len(nmads[nmads>3])
    return outs

def tempsil(t,x,l,s=200,kn=200,c=1):
    # Description: implementation of the Temporal Silhouette index for the
    # internal validation of streaming clustering, FIV, Jun 2022
    #
    # INPUTS
    # t: 1D-array with timestamps
    # x: XD-array with data vectors
    # l: 1D-array with labels
    # s: window-size of the simple-moving-average (SMA)
    # kn: number-of-neighbors of other clusters for calculating beta
    # c: sigma parameter to weight the penalization over contextual outliers [0...1]
    #
    # OUTPUTS
    # k: 1D-array with cluster-labels
    # ts2: 1D-array with quadratic cluster temporal silhouettes
    # TS: global Temporal Silhuette
 
    k = np.unique(l)
    ts = np.zeros(len(k))
    for i,label in enumerate(k):
        tl0 = t[l==label]
        tl1 = np.roll(tl0,-1)
        dtl = (tl1-tl0)[:-1]
        xk = x[l==label,:]
        IAD = 1 / (1 + c * mad(dtl,True)/xk.shape[0])
        wj = np.argwhere(l==label)
        wnt = np.argwhere(l!=label)
        SMA = np.zeros(xk.shape)
        tst = np.zeros(xk.shape)
        for a in range(xk.shape[1]):
            SMA[:,a] = pd.Series(xk[:,a]).rolling(s,min_periods=1, center=True).mean().to_numpy()
        a = np.zeros(len(SMA))
        b = np.zeros(len(SMA))
        for j in range(len(SMA)-1):
            a[j] =  distance.euclidean(xk[j],SMA[j])
            tst[j] = 0
            if len(wnt)>0:
                m = find_knearest(wnt,wj[j],kn,option='nearest')
                b[j] = dist2oc([xk[j]],x[m,:],l[m])
                tst[j] = (b[j] - a[j])/np.max([a[j],b[j]])
        TCD = temp_centroid_coherence(SMA)/xk.shape[0]
        ts[i] = (1 + np.mean(tst)) * IAD * (1 - TCD) - 1
    _,card = np.unique(l,return_counts=True)
    ts2 = np.power(ts,2) * np.sign(ts)
    TS2 = np.sum(card*ts2)/np.sum(card)
    TS = np.sqrt(np.abs(TS2)) * np.sign(TS2)
    return k,ts2,TS

# def calculate_ssq(data_points, centroids, labels):
#     """
#     Calculate the Sum of Squared Distances (SSQ).

#     Parameters:
#     - data_points: A 2D numpy array where each row represents a data point.
#     - centroids: A 2D numpy array where each row represents a cluster centroid.
#     - labels: A 1D numpy array where each entry represents the cluster index for the corresponding data point.

#     Returns:
#     - SSQ: The sum of squared distances from each data point to its assigned centroid.
#     """
#     ssq = 0.0
#     unique_labels = list(set(labels))
#     data_points['cluster'] = labels

#     for cluster in unique_labels:
#         squared_distance = 0
#         # Get the assigned cluster centroid
#         centroid = centroids.loc[cluster].values
#         points = data_points[data_points['cluster']==cluster].drop(['cluster'], axis=1).values
#         # Calculate the squared distance between the data point and the centroid
#         for p in points:
#             squared_distance += np.sum((p - centroid) ** 2)
#         # Add to the total SSQ
#         ssq += squared_distance
    
#     return ssq

# import numpy as np
# import pandas as pd

def calculate_ssq(data_points, centroids, labels):
    """
    Calculate the Sum of Squared Distances (SSQ).

    Parameters:
    - data_points: A pandas DataFrame where each row represents a data point.
    - centroids: A pandas DataFrame where each row represents a cluster centroid.
    - labels: A 1D numpy array where each entry represents the cluster index for the corresponding data point.

    Returns:
    - SSQ: The sum of squared distances from each data point to its assigned centroid.
    """
    ssq = 0.0
    data_points = data_points.copy()  # Avoid modifying the original DataFrame
    data_points['cluster'] = labels

    unique_labels = data_points['cluster'].unique()

    for cluster in unique_labels:
        # Get the assigned cluster centroid
        centroid = centroids.loc[cluster].values
        # Get points in this cluster
        points = data_points[data_points['cluster'] == cluster].drop(columns=['cluster']).values
        # Calculate the squared distance between the data points and the centroid
        squared_distance = np.sum((points - centroid) ** 2)
        # Add to the total SSQ
        ssq += squared_distance

    return ssq



def calculate_ssq(data_points, centroids, labels):
    """
    Calculate the Sum of Squared Distances (SSQ).

    Parameters:
    - data_points: A pandas DataFrame where each row represents a data point.
    - centroids: A pandas DataFrame where each row represents a cluster centroid.
    - labels: A 1D numpy array where each entry represents the cluster index for the corresponding data point.

    Returns:
    - SSQ: The sum of squared distances from each data point to its assigned centroid.
    """
    ssq = 0.0
    data_points = data_points.copy()  # Avoid modifying the original DataFrame
    data_points['cluster'] = labels

    unique_labels = data_points['cluster'].unique()
    _data_points = data_points.drop(['cluster'], axis=1).values
    for i in range(len(_data_points)):
        min_distance = 1.7976931348623157e+308
        for c in range(len(unique_labels)):
            distance = 0
            centroid = centroids.loc[unique_labels[c]].values
            # for p in range(len(centroid)):
            #     d = _data_points[i][p] - centroid[p]
            #     distance += d * d

            distance = np.sum((_data_points[i] - centroid) ** 2)
            min_distance = min(distance, min_distance)
        ssq+=min_distance

    return ssq
