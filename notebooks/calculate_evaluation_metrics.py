from sklearn.metrics import calinski_harabasz_score, pairwise_distances, classification_report
from sklearn.metrics import silhouette_score, rand_score, adjusted_rand_score
import pandas as pd
import numpy as np

window_size = 32
expname = 'kddcup'
print(f"Window size: {window_size}")
df = pd.read_csv(f"/root/EdgeCluster/results/{expname}/tabular/{window_size}/final_clustering.csv")
# df = df[df['cluster']!=-1]

target = df["target"]
cluster = df["cluster"]
timestamps=df["ids"]
df = df[df.columns[:-6]]

def calculate_distances(data):
    return pairwise_distances(data,data)

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

# F1
def f_measure(pred, true):
    value = (2 * len(pred & true)) / (len(true) + len(pred))
    return value

# SI
def calculate_silhouette(data, clusters):
    # distances = calculate_distances(data.to_numpy(copy=True))
    # distances = calculate_distances_improved(data.to_numpy(copy=True))
    return silhouette_score(data, clusters, n_jobs=-1)

from scipy.spatial import distance
from scipy.stats import median_abs_deviation
import numpy as np

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


def evalutation_report(data, timestamps, pred_labels, true_labels):
    # distances = calculate_distances_improved(data.to_numpy(copy=True))
    # connectivity = calculate_connectivity(data, 
    #                                     pred_labels,
    #                                     [x for x in range(data.shape[1])],
    #                                     10, distance_matrix=distances)['CONN'].sum()
    
    # F1 = f_measure(pred_labels, true_labels)
    # RI = rand_score(true_labels, pred_labels)
    # ARI = adjusted_rand_score(true_labels, pred_labels)
    if len(pred_labels.unique()) > 1:
        SI = calculate_silhouette(data, pred_labels)
        # SI = None
        # _,coeff, TSI = tempsil(timestamps.values, data.values, cluster.values)
        # s = calinski_harabasz_score(data, pred_labels)
    else:
        SI = None
        TSI = None
        s = None
        F1 = None
        RI = None
        ARI = None

    return None, SI, None, None, None, None

F1, SI, TSI, s, RI, ARI = evalutation_report(data=df, timestamps=timestamps, pred_labels=cluster, true_labels=target)

metrics_df =  pd.DataFrame({
    'F1': [F1],
    'SI': [SI],
    'TSI': [TSI],
    's': [s],
    'RI': [RI],
    'ARI': [ARI]
})
print(metrics_df.head())
metrics_df.to_csv(f"output_{window_size}_{expname}_with_-1-second.csv")