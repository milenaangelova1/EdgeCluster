import numpy as np

def calculate_ssq(data_points, centroids, labels):
    """
    Calculate the Sum of Squared Distances (SSQ).

    Parameters:
    - data_points: A 2D numpy array where each row represents a data point.
    - centroids: A 2D numpy array where each row represents a cluster centroid.
    - labels: A 1D numpy array where each entry represents the cluster index for the corresponding data point.

    Returns:
    - SSQ: The sum of squared distances from each data point to its assigned centroid.
    """
    ssq = 0.0
    for i in range(len(data_points)):
        # Get the assigned cluster centroid
        centroid = centroids[labels[i]]
        # Calculate the squared distance between the data point and the centroid
        squared_distance = np.sum((data_points[i] - centroid) ** 2)
        # Add to the total SSQ
        ssq += squared_distance
    
    return ssq

# Example usage:
data_points = np.array([[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0]])
centroids = np.array([[1.25, 1.9], [6.5, 8.0]])
labels = np.array([0, 0, 1, 1]) 

ssq = calculate_ssq(data_points, centroids, labels)
print(f"Sum of Squared Distances (SSQ): {ssq}")