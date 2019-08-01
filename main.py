from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import numpy as np
from util.constants import Constants
from business.main_functions import MainFunctions
from util.plot_util import PlotUtil
from business.metrics_functions import MetricsFunctions
from util.file_util import FileUtil
import pandas as pd


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
        self.__option = MainFunctions.KMEANS
        self.__filename_array = [MainFunctions.KMEANS, MainFunctions.PCA_KMEANS, MainFunctions.TSNE_KMEANS,
                                 MainFunctions.PSO_KMEANS, MainFunctions.PSO_PCA_KMEANS]

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
        #self.__execute_metrics(kmeans_array=kmeans_array)
        self.__print_metrics()
        #PlotUtil.generate_boxplot_by_file(self.__filename_array)

    def __execute_metrics(self, kmeans_array):
        array_data = [MetricsFunctions.execute_silhouette(kmeans_array=kmeans_array, data=self.__data),
                      MetricsFunctions.execute_v_measure_score(kmeans_array=kmeans_array, labels=self.__labels),
                      MetricsFunctions.execute_adjusted_rand_score(kmeans_array=kmeans_array, labels=self.__labels),
                      MetricsFunctions.execute_calinski_harabasz_score(kmeans_array=kmeans_array, data=self.__data)]
        array_metrics_names = [MetricsFunctions.SILHOUETTE, MetricsFunctions.V_MEASURE_SCORE,
                               MetricsFunctions.ADJUSTED_RAND_SCORE, MetricsFunctions.CALINSKI_HARABASZ_SCORE]
        FileUtil.create_file(array_data=array_data, array_metrics_names=array_metrics_names, filename=self.__option)

    def __print_metrics(self):
        mean_metrics = PlotUtil.get_values_to_print_mean_metrics(self.__filename_array)
        data_function = []
        size = len(self.__filename_array)
        mean_size = len(mean_metrics)
        for i in range(size):
            data = []
            for j in range(mean_size):
                if j == 0:
                    index = i
                else:
                    index = i + (j*size)
                    if index >= mean_size:
                        break
                data.append(mean_metrics[index])
            data_function.append(data)

        df = pd.DataFrame(data=data_function, columns=MetricsFunctions.ALL_METRICS, index=self.__filename_array)
        PlotUtil.generate_table_by_dataframe(df)


main = Main()
main.execute()
