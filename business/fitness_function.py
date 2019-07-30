from sklearn.cluster import KMeans
from sklearn import metrics
from util.constants import Constants
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class FitnessFunction:

    @staticmethod
    @ignore_warnings(category=ConvergenceWarning)
    def run(position, data):
            kmeans = KMeans(init=position, n_clusters=Constants.N_CLUSTERS, n_init=1)
            kmeans.fit(data)
            return metrics.calinski_harabasz_score(data, kmeans.labels_)
