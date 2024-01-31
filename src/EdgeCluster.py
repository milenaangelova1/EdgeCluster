from utils import (dist, 
                   recalculate_cluster_params,
                   recalculate_cluster_vectors, 
                   find_the_closest_cluster, 
                   update_initial_clustering,
                   remove_cluster_from_clustering,
                   remove_closed_cluster_metrics)

class EdgeCluster:
    """
    Edge Cluster Algorithm
    """
    def __init__(self) -> None:
        pass

    def fit(self, window:dict, initial_clustering:list):
        """
        Adapts the newly incomming data to the offline clustering.

        :param: window - a dict - It represents the current processed window. The list contains the high, low, and mean vectors.
        :param: initial_clustring - a list of dicts. It represents the offline clustering (initial clustering). 
                Each tuple contains the high, low, and mean vectors.
        """
        matched, deviated, merges, windows, merge_with_window = False, False, [], [], False
        cluster = find_the_closest_cluster(window, initial_clustering)
        if dist(cluster['mean'], window['mean']) > (dist(cluster['low'], cluster['mean']) + dist(window['mean'], window['high'])):
            # D(m_i, m_w) > (D(l_i, m_i) + D(m_w, h_w))
            deviated = True
        
        elif (dist(cluster['mean'], window['mean']) < dist(cluster['low'], cluster['mean'])) and (dist(cluster['mean'], window['mean']) < dist(window['mean'], window['high'])):
            # (D(m_i, m_w) < D(l_i, m_i)) and (D(m_i, m_w) < D(m_w, h_w))
            new_cluster = recalculate_cluster_vectors(cluster, window)
            update_initial_clustering(new_cluster, initial_clustering)
            matched = True

        elif (dist(cluster['low'], cluster['mean']) >= dist(cluster['mean'], window['mean'])) or (dist(cluster['mean'], window['mean']) <= dist(window['mean'], window['high'])):
            # (D(l_i, m_i) >= D(m_i, m_w)) or (D(m_i, m_w) <= D(m_w, h_w))
            # merging c and w and recalculating the window - low, high, and mean vectors.
            temp_cluster = cluster.copy()
            new_cluster = recalculate_cluster_params(temp_cluster, window)
           
            update_initial_clustering(new_cluster, initial_clustering)
            cluster = temp_cluster
            
            # temp_clustering = initial_clustering.copy()
            # temp_clustering = remove_cluster_from_clustering(temp_clustering, cluster_for_removing = new_cluster)
            
            # while True:
            # closed_cluster = find_the_closest_cluster(new_cluster, temp_clustering)

            # if closed_cluster is not None:

            #     # if not bool(closed_cluster):
            #     #     break

            #     if new_cluster['cluster'] != closed_cluster['cluster'] and ((dist(new_cluster['low'], closed_cluster['mean']) >= dist(new_cluster['mean'], closed_cluster['mean'])) or (dist(new_cluster['mean'], closed_cluster['mean']) <= dist(closed_cluster['mean'], closed_cluster['high']))):
            #         # ((D(l_i, m_j ) >= D(m_i, m_j )) or (D(m_i, m_j ) <= D(m_j , h_j))
            #         # merge clusters c and j. Merging will be union
            #         new_next_cluster = recalculate_cluster_params(new_cluster, closed_cluster)
            #         remove_closed_cluster_metrics(closed_cluster, initial_clustering)
            #         # find and change the cluster into initial clustering solution
            #         update_initial_clustering(new_next_cluster, initial_clustering)
            #         merges.append(closed_cluster['cluster'])
            #     else:
            #         merge_with_window = True
            # else:
            merge_with_window = True

        elif (dist(cluster['low'], cluster['mean']) < dist(cluster['mean'], window['mean'])) and (dist(cluster['mean'], window['mean']) > dist(window['mean'], window['high'])):
            # (D(l_i, m_i) < D(m_i, m_w)) and (D(m_i, m_w) > D(m_w, h_w))
            # the merging will be union between both
            sorted_results = sorted(initial_clustering, key=lambda x: x['cluster'], reverse=True)
            cluster_label = sorted_results[0]['cluster'] + 1
            window['cluster'] = cluster_label
            windows.append(cluster_label)
            initial_clustering.append(window)

        return {
            "clustering":initial_clustering, 
            "closed_cluster": cluster, 
            "changes": {
                "deviated": deviated,
                "matched": matched,
                "merges": merges,
                "windows": windows,
                "merge_with_window": merge_with_window
            },
            "window": window
        }