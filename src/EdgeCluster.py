from utils import dist, recalculate_window_params, find_the_closest_cluster, update_initial_clustering, remove_cluster_from_clustering

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
        matched, deviated, merges, windows = False, False, [], []
        cluster = find_the_closest_cluster(window, initial_clustering)
        if dist(cluster['mean'], window['mean']) > (dist(cluster['low'], cluster['mean']) + dist(window['mean'], window['high'])):
            # D(m_i, m_w) > (D(l_i, m_i) + D(m_w, h_w))
            deviated = True
            return {
                "clustering":initial_clustering, 
                "closed_cluster": cluster, 
                "changes": {
                    "deviated": deviated,
                    "matched": matched,
                    "merges": merges,
                    "windows": windows
                },
                "window": window
            }
        elif (dist(cluster['mean'], window['mean']) < dist(cluster['low'], cluster['mean'])) and (dist(cluster['mean'], window['mean']) < dist(window['mean'], window['high'])):
            # (D(m_i, m_w) < D(l_i, m_i)) and (D(m_i, m_w) < D(m_w, h_w))
            matched = True
            return {
                "clustering":initial_clustering, 
                "closed_cluster": cluster, 
                "changes": {
                    "deviated": deviated,
                    "matched": matched,
                    "merges": merges,
                    "windows": windows
                },
                "window": window
            }
        elif (dist(cluster['low'], cluster['mean']) >= dist(cluster['mean'], window['mean'])) or (dist(cluster['mean'], window['mean']) <= dist(window['mean'], window['high'])):
            # (D(l_i, m_i) >= D(m_i, m_w)) or (D(m_i, m_w) <= D(m_w, h_w))
            # merging c and w and recalculating the window - low, high, and mean vectors.
            cluster = recalculate_window_params(cluster, window)
            for j in initial_clustering:
                if cluster['cluster'] != j['cluster'] and (dist(cluster['low'], j['mean']) >= dist(cluster['mean'], j['mean'])) or (dist(cluster['mean'], j['high']) <= dist(j['mean'], j['high'])):
                        # ((D(l_i, m_j ) >= D(m_i, m_j )) or (D(m_i, m_j ) <= D(m_j , h_j))
                        # merge clusters c and j. Merging will be union
                        cluster = recalculate_window_params(cluster, window)
                        # find and change the cluster into initial clustering solution
                        update_initial_clustering(cluster, initial_clustering)
                        remove_cluster_from_clustering(initial_clustering, cluster_for_removing = j)
                        merges.append(j['cluster'])
        elif (dist(cluster['low'], cluster['mean']) < dist(cluster['mean'], window['mean'])) and (dist(cluster['mean'], window['mean']) > dist(window['mean'], window['high'])):
            # (D(l_i, m_i) < D(m_i, m_w)) and (D(m_i, m_w) > D(m_w, h_w))
            # merge c with w
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
                "windows": windows
            },
            "window": window
        }


