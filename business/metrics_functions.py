from sklearn import metrics
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score
from util.constants import Constants


class MetricsFunctions:
    SILHOUETTE = "silhouette"
    V_MEASURE_SCORE = "v_measure_score"
    ADJUSTED_RAND_SCORE = "adjusted_rand_score"
    CALINSKI_HARABASZ_SCORE = "calinski_harabasz_score"
    ALL_METRICS = [SILHOUETTE, V_MEASURE_SCORE, ADJUSTED_RAND_SCORE, CALINSKI_HARABASZ_SCORE]

    @staticmethod
    def execute_silhouette(kmeans_array, data):
        silhouette_array = []
        for kmeans in kmeans_array:
            silhouette_array.append(metrics.silhouette_score(data, kmeans.labels_, metric='euclidean',
                                                             sample_size=Constants.SAMPLE_SIZE))
        return silhouette_array

    @staticmethod
    def execute_v_measure_score(kmeans_array, labels):
        v_measure_score_array = []
        for kmeans in kmeans_array:
            v_measure_score_array.append(v_measure_score(labels, kmeans.labels_))
        return v_measure_score_array

    @staticmethod
    def execute_adjusted_rand_score(kmeans_array, labels):
        adjusted_rand_score_array = []
        for kmeans in kmeans_array:
            adjusted_rand_score_array.append(adjusted_rand_score(labels, kmeans.labels_))
        return adjusted_rand_score_array

    @staticmethod
    def execute_calinski_harabasz_score(kmeans_array, data):
        calinski_harabasz_score_array = []
        for kmeans in kmeans_array:
            calinski_harabasz_score_array.append(metrics.calinski_harabasz_score(data, kmeans.labels_))
        return calinski_harabasz_score_array
