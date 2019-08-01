from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import numpy as np
from util.constants import Constants
from business.main_functions import MainFunctions
from util.plot_util import PlotUtil
from business.metrics_functions import MetricsFunctions
from util.file_util import FileUtil


class Main:

    def __init__(self):
        self.__load_database()
        self.__sample_size = 300
        self.__pso = None
        self.__options = {"kmeans": MainFunctions.execute_kmeans,
                          "pca_kmeans": MainFunctions.execute_pca_kmeans,
                          "tsne_kmeans": MainFunctions.execute_tsne_kmeans,
                          "pso_kmeans": MainFunctions.execute_pso_kmeans,
                          "pso_pca_kmeans": MainFunctions.execute_pso_pca_kmeans,
                          "tsne_pso_kmeans": MainFunctions.execute_tsne_pso_kmeans,
                          "tsne_pca_pso_kmeans": MainFunctions.execute_tsne_pca_pso_kmeans
                          }
        self.__option = MainFunctions.TSNE_KMEANS
        self.__filename_array = [MainFunctions.KMEANS, MainFunctions.PCA_KMEANS]

    def __load_database(self):
        """Load the database"""
        digits = load_digits()
        self.__data = scale(digits.data)
        n_samples, self.__n_features = self.__data.shape
        self.__n_digits = len(np.unique(digits.target))
        self.__labels = digits.target
        print("n_digits: %d, \t n_samples %d, \t n_features %d"
              % (self.__n_digits, n_samples, self.__n_features))

    def execute(self):
        kmeans_array = []
        for i in range(Constants.N_ITERATIONS):
            print("Iteration ", i + 1)
            kmeans = self.__options[self.__option](n_clusters=self.__n_digits, data=self.__data)
            kmeans_array.append(kmeans)
        print("[Generating results]")
        self.__execute_metrics(kmeans_array=kmeans_array)
        PlotUtil.generate_boxplot_by_file(self.__filename_array)

    def __execute_metrics(self, kmeans_array):
        array_data = [MetricsFunctions.execute_silhouette(kmeans_array=kmeans_array, data=self.__data),
                      MetricsFunctions.execute_v_measure_score(kmeans_array=kmeans_array, labels=self.__labels),
                      MetricsFunctions.execute_adjusted_rand_score(kmeans_array=kmeans_array, labels=self.__labels),
                      MetricsFunctions.execute_calinski_harabasz_score(kmeans_array=kmeans_array, data=self.__data)]
        array_metrics_names = [MetricsFunctions.SILHOUETTE, MetricsFunctions.V_MEASURE_SCORE,
                               MetricsFunctions.ADJUSTED_RAND_SCORE, MetricsFunctions.CALINSKI_HARABASZ_SCORE]
        FileUtil.create_file(array_data=array_data, array_metrics_names=array_metrics_names, filename=self.__option)

    def __compare_results(self, kmeans_results, pca_kmeans_results, tsne_kmeans_results, pso_kmeans_results,
                          pso_pca_kmeans_results, tsne_pso_kmeans_results, tsne_pca_pso_kmeans_results):
        print(82 * '_')
        self.__print_metrics("K-means", kmeans_results)
        self.__print_metrics("PCA + K-means", pca_kmeans_results)
        self.__print_metrics("t-SNE + K-means", tsne_kmeans_results)
        self.__print_metrics("PSO K-means", pso_kmeans_results)
        self.__print_metrics("PSO PCA Kmeans", pso_pca_kmeans_results)
        self.__generate_boxplot(kmeans_results, pca_kmeans_results, tsne_kmeans_results, pso_kmeans_results,
                                pso_pca_kmeans_results, tsne_pso_kmeans_results, tsne_pca_pso_kmeans_results)
        print(82 * '_')

    def __print_metrics(self, title, kmeans_array):
        silhouette_total = np.sum(self.__execute_silhouette(kmeans_array))
        v_measure_score_total = np.sum(self.__execute_v_measure_score(kmeans_array))
        adjusted_rand_score_total = np.sum(self.__execute_adjusted_rand_score(kmeans_array))
        calinski_harabasz_score_total = np.sum(self.__execute_calinski_harabasz_score(kmeans_array))

        silhouette_mean = silhouette_total / Constants.N_ITERATIONS
        v_measure_score_mean = v_measure_score_total / Constants.N_ITERATIONS
        adjusted_rand_score_mean = adjusted_rand_score_total / Constants.N_ITERATIONS
        calinski_harabasz_score_mean = calinski_harabasz_score_total / Constants.N_ITERATIONS

        print('\t\tsilhouette\tv_measure_score\tadjusted_rand_score\tcalinski_harabasz_score')
        print(title, '\t%.3f' % silhouette_mean, '\t%.3f' %
              v_measure_score_mean, '\t\t\t%.3f' %
              adjusted_rand_score_mean, '\t\t\t%.3f' %
              calinski_harabasz_score_mean)


main = Main()
main.execute()
