from utils import dist, recalculate_window_params, find_the_closest_cluster, update_initial_clustering, convert_to_list, draw_graph, remove_cluster_from_clustering
import pandas as pd

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
        matched, deviated, merges = False, False, []
        if not initial_clustering:
            return
        window_metrics = window['cluster_metrics']
        cluster = find_the_closest_cluster(window_metrics, convert_to_list(initial_clustering))
        if dist(cluster['mean'], window_metrics['mean']) > (dist(cluster['low'], cluster['mean']) + dist(window_metrics['mean'], window_metrics['high'])):
            # D(m_i, m_w) > (D(l_i, m_i) + D(m_w, h_w))
            deviated = True
            return {
                "clustering":initial_clustering, 
                "closed_cluster": cluster, 
                "changes": {
                    "deviated": deviated,
                    "matched": matched,
                    "merges": merges
                },
                "window": window_metrics
            }
        elif (dist(cluster['mean'], window_metrics['mean']) < dist(cluster['low'], cluster['mean'])) and (dist(cluster['mean'], window_metrics['mean']) < dist(window_metrics['mean'], window_metrics['high'])):
            # (D(m_i, m_w) < D(l_i, m_i)) and (D(m_i, m_w) < D(m_w, h_w))
            matched = True
            return {
                "clustering":initial_clustering, 
                "closed_cluster": cluster, 
                "changes": {
                    "deviated": deviated,
                    "matched": matched,
                    "merges": merges
                },
                "window": window_metrics
            }
        elif (dist(cluster['low'], cluster['mean']) >= dist(cluster['mean'], window_metrics['mean'])) or (dist(cluster['mean'], window_metrics['mean']) <= dist(window_metrics['mean'], window_metrics['high'])):
            # (D(l_i, m_i) >= D(m_i, m_w)) or (D(m_i, m_w) <= D(m_w, h_w))
            # merging c and w and recalculating the window - low, high, and mean vectors.
            cluster = recalculate_window_params(cluster, window_metrics)
            for j in initial_clustering:
                if cluster['cluster'] != j['cluster'] and (dist(cluster['low'], j['mean']) >= dist(cluster['mean'], j['mean'])) or (dist(cluster['mean'], j['high']) <= dist(j['mean'], j['high'])):
                        # ((D(l_i, m_j ) >= D(m_i, m_j )) or (D(m_i, m_j ) <= D(m_j , h_j))
                        # merge clusters c and j. Merging will be union
                        cluster = recalculate_window_params(cluster, window_metrics)
                        # find and change the cluster into initial clustering solution
                        update_initial_clustering(cluster, initial_clustering)
                        remove_cluster_from_clustering(initial_clustering, cluster_for_removing = j)
                        merges.append(j['cluster'])
        elif (dist(cluster['low'], cluster['mean']) < dist(cluster['mean'], window_metrics['mean'])) and (dist(cluster['mean'], window_metrics['mean']) > dist(window_metrics['mean'], window_metrics['high'])):
            # (D(l_i, m_i) < D(m_i, m_w)) and (D(m_i, m_w) > D(m_w, h_w))
            # merge c with w
            # the merging will be union between both
            sorted_results = sorted(initial_clustering, key=lambda x: x['cluster'], reverse=True)
            cluster_label = sorted_results[0]['cluster'] + 1
            initial_clustering.append({
                'high': window_metrics['high'],
                'low': window_metrics['low'],
                'mean': window_metrics['mean'],
                'cluster': cluster_label,
                'stream': window_metrics['stream']
            })

        return {
            "clustering":initial_clustering, 
            "closed_cluster": cluster, 
            "changes": {
                "deviated": deviated,
                "matched": matched,
                "merges": merges
            },
            "window": window
        }


