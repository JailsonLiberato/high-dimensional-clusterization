from sklearn.cluster import KMeans
from sklearn import metrics
from util.constants import Constants
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np


class FitnessFunction:

    @staticmethod
    @ignore_warnings(category=ConvergenceWarning)
    def run(position, data):
        kmeans = KMeans(init=position, n_clusters=Constants.N_CLUSTERS, n_init=1)
        kmeans.fit(data)
        distances = 0
        labels = kmeans.labels_
        for i, c in enumerate(position):
            idx = np.where(labels == i)
            dist = np.sum((data[idx] - c) ** 2)
            distances += dist
        return distances


