from utils import get_data_windows, get_initial_clustering, dist, recalculate_window_params, find_the_closest_cluster


class EdgeCluster:
    """
    Edge Cluster Algorithm
    """

    def fit(self, w:dict, C:list):
        """
        Adapts the newly incomming data to the offline clustering.

        :param: w - a dict - It represents the current processed window. The list contains the high, low, and mean vectors.
        :param: C - a list of dicts. It represents the offline clustering (initial clustering). 
                Each tuple contains the high, low, and mean vectors.
        """

        c = find_the_closest_cluster(w, C)
        if dist(c['mean'], w['mean']) > (dist(c['low'], c['mean']) + dist(w['mean'], w['high'])):
            # D(m_i, m_w) > (D(l_i, m_i) + D(m_w, h_w))
            return C
        elif (dist(c['mean'], w['mean']) < dist(c['low'], c['mean'])) and (dist(c['mean'], w['mean']) < dist(w['mean'], w['high'])):
            # (D(m_i, m_w) < D(l_i, m_i)) and (D(m_i, m_w) < D(m_w, h_w))
            return C
        elif (dist(c['low'], c['mean']) >= dist(c['mean'], w['mean'])) or (dist(c['mean'], w['mean']) <= dist(w['mean'], w['high'])):
            # (D(l_i, m_i) >= D(m_i, m_w)) or (D(m_i, m_w) <= D(m_w, h_w))
            # merging c and w and recalculating the window - low, high, and mean vectors.
            c = recalculate_window_params(c, w)
            for j in C:
                if c != j:
                    if (dist(c['low'], j['mean']) >= dist(c['mean'], j['mean'])) or (dist(c['mean'], j['high']) <= dist(j['mean'], j['high'])):
                        # ((D(l_i, m_j ) >= D(m_i, m_j )) or (D(m_i, m_j ) <= D(m_j , h_j))
                        # merge clusters c and j. Merging will be union
                        c = recalculate_window_params(c, w)
        elif (dist(c['low'], c['mean']) < dist(c['mean'], w['mean'])) and (dist(c['mean'], w['mean']) > dist(w['mean'], w['high'])):
            # (D(l_i, m_i) < D(m_i, m_w)) and (D(m_i, m_w) > D(m_w, h_w))
            # merge c with w
            # the merging will be union between both
            C = C.add(w)
        return C
        


