from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics
import numpy as np


class Main:

    def __init__(self):
        self.__load_database()
        self.__sample_size = 300

    def __load_database(self):
        """Load the database"""
        digits = load_digits()
        self.__data = scale(digits.data)
        n_samples, self.__n_features = self.__data.shape
        self.__n_digits = len(np.unique(digits.target))
        labels = digits.target
        print("n_digits: %d, \t n_samples %d, \t n_features %d"
              % (self.__n_digits, n_samples, self.__n_features))
        print(labels)

    def execute(self):
        """Execute all algorithms"""
        kmeans_results = self.__execute_kmeans()
        pso_kmeans_results = self.__execute_pso_kmeans()
        pca_kmeans_results = self.__execute_pca_kmeans()
        pca_pso_kmeans_results = self.__execute_pca_pso_kmeans()
        tsne_kmeans_results = self.__execute_tsne_kmeans()
        tsne_pso_kmeans_results = self.__execute_tsne_pso_kmeans()
        self.__compare_results(kmeans_results, pso_kmeans_results, pca_kmeans_results, pca_pso_kmeans_results,
                               tsne_kmeans_results, tsne_pso_kmeans_results)

    def __execute_kmeans(self):
        kmeans = KMeans(n_clusters=self.__n_digits)
        kmeans.fit(self.__data)
        return kmeans

    def __execute_pso_kmeans(self):
        pso = ParticleSwarmOptimization(
            n_cluster=10, n_particles=10, data=self.__data, hybrid=False, max_iter=2000, print_debug=2000)
        pso.run()

    def __execute_pca_kmeans(self):
        pass

    def __execute_pca_pso_kmeans(self):
        pass

    def __execute_tsne_kmeans(self):
        pass

    def __execute_tsne_pso_kmeans(self):
        pass

    def __compare_results(self, kmeans_results, pso_kmeans_results, pca_kmeans_results, pca_pso_kmeans_results,
                          tsne_kmeans_results, tsne_pso_kmeans_results):
        print(82 * '_')
        print('\t\tsilhouette')
        print('K-means\t\t%.3f' % metrics.silhouette_score(self.__data, kmeans_results.labels_, metric='euclidean',
                                                           sample_size=self.__sample_size))
        print('PSO + K-means\t\t%.3f' % metrics.silhouette_score(self.__data, pso_kmeans_results.labels_,
                                                                 metric='euclidean', sample_size=self.__sample_size))
        print(82 * '_')


main = Main()
main.execute()
