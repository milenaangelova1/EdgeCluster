import numpy as np
from sklearn.metrics import pairwise_distances

def kmedoids(D, k, medoids=None, tmax=100):
    '''
        D is the dissimilarity matrix of the data to be clustered
        in a n x n format. 
        
        k is the amount of clusters.
        
        medoids is the initial medoids chosen for the algorithm,
        if none is passed we initialize with random ones below
        
        tmax is the maximum amount of iterations for the alg
    '''
    # Dimensions of dissimilarity matrix
    m,n = D.shape
    
    # Random initation of medoid indices if none is passed
    if medoids is None:
        medoids = np.sort(np.random.choice(n, k))
    new_medoids = medoids.copy()
    medoids_copy = medoids.copy()
    np.sort(medoids_copy)
    
    # Initial dict for our clusters
    C = {}
    
    for t in range(tmax):
        # Determine the clusters
        J = np.argmin(D[:,medoids], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        
        # Update the cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])], axis=1)
            j = np.argmin(J)
            new_medoids[kappa] = C[kappa][j]
        new_medoids_copy = new_medoids.copy()
        np.sort(new_medoids_copy)
        
        # check for convergence, i.e. are the matrices the same?
        # if so, then we are finished
        if np.array_equal(medoids_copy, new_medoids_copy):
            break
            
        medoids = new_medoids.copy()
    
    # If we didn't break the loop above, then we have to update the
    # cluster memberships again
    else:
        J = np.argmin(D[:,medoids], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    
    return C, medoids

def calculate_distances(data, metric):
    return pairwise_distances(data,data, metric)

def pairwise_hamming(data):
    return pairwise_distances(data,data, metric='hamming')

def pairwise_euclidean(data):
    return pairwise_distances(data,data, metric='euclidean')
