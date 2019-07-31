from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from business.pso import ParticleSwarmOptimization
from util.constants import Constants


class Main:

    def __init__(self):
        self.__load_database()
        self.__sample_size = 300
        self.__pso = None

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
        """Execute all algorithms"""
        kmeans_results = []
        pca_kmeans_results = []
        tsne_kmeans_results = []
        pso_kmeans_results = []
        pso_pca_kmeans_results = []
        tsne_pso_kmeans_results = []
        tsne_pca_pso_kmeans_results = []
        for i in range(Constants.N_ITERATIONS):
            print("Iteration ", i + 1)
            kmeans_results.append(self.__execute_kmeans())
            pca_kmeans_results.append(self.__execute_pca_kmeans())
            tsne_kmeans_results.append(self.__execute_tsne_kmeans())
            pso_kmeans_results.append(self.__execute_pso_kmeans())
            pso_pca_kmeans_results.append(self.__execute_pso_pca_kmeans())
            tsne_pso_kmeans_results.append(self.__execute_tsne_pso_kmeans())
            tsne_pca_pso_kmeans_results.append(self.__execute_tsne_pca_pso_kmeans())
        print("[Generating results]")
        self.__compare_results(kmeans_results, pca_kmeans_results, tsne_kmeans_results, pso_kmeans_results,
                               pso_pca_kmeans_results, tsne_pso_kmeans_results, tsne_pca_pso_kmeans_results)
        self.__generate_boxplot(kmeans_results, pca_kmeans_results, tsne_kmeans_results, pso_kmeans_results,
                                pso_pca_kmeans_results, tsne_pso_kmeans_results, tsne_pca_pso_kmeans_results)

    def __execute_kmeans(self):
        print("Executing K-means...")
        kmeans = KMeans(n_clusters=self.__n_digits)
        kmeans.fit(self.__data)
        return kmeans

    def __execute_pca_kmeans(self):
        print("Executing PCA K-means...")
        pca = PCA(n_components=self.__n_digits).fit(self.__data)
        kmeans = KMeans(init=pca.components_, n_clusters=self.__n_digits, n_init=1)
        kmeans.fit(self.__data)

        return kmeans

    def __execute_tsne_kmeans(self):
        print("Executing t-SNE K-means...")
        tsne = TSNE(n_components=2).fit_transform(self.__data)
        kmeans_tsne = KMeans(n_clusters=self.__n_digits, random_state=0)
        kmeans_tsne.fit_predict(tsne)
        return kmeans_tsne

    def __execute_pso(self, title, gbest_initial=None):
        self.__pso = ParticleSwarmOptimization(n_dimensions=self.__n_features, data=self.__data,
                                               gbest_initial=gbest_initial)
        self.__pso.optimize()
        self.__generate_plot(title + " Convergency", self.__pso.gbest_fitness_array)

    def __execute_pso_kmeans(self):
        print("Executing PSO K-means...")
        self.__execute_pso(title="PSO Kmeans")
        kmeans = KMeans(init=self.__pso.gbest, n_clusters=self.__n_digits, n_init=1)
        kmeans.fit(self.__data)
        return kmeans

    def __execute_pso_pca_kmeans(self):
        print("Executing PSO PCA K-means...")
        pca = PCA(n_components=self.__n_digits, ).fit(self.__data)
        self.__execute_pso(title="PSO PCA Kmeans", gbest_initial=pca.components_)
        kmeans = KMeans(init=self.__pso.gbest, n_clusters=self.__n_digits, n_init=1)
        kmeans.fit(self.__data)
        return kmeans

    def __execute_tsne_pso_kmeans(self):
        print("Executing t-SNE PSO K-means...")
        self.__execute_pso(title="tSNE PSO Kmeans")
        tsne_pso = TSNE(n_components=2).fit_transform(self.__pso.gbest)
        tsne = TSNE(n_components=2,  init=tsne_pso).fit_transform(self.__data)
        kmeans_tsne = KMeans(n_clusters=self.__n_digits, random_state=0)
        kmeans_tsne.fit_predict(tsne)
        return kmeans_tsne

    def __execute_tsne_pca_pso_kmeans(self):
        print("Executing t-SNE PCA PSO K-means...")
        pca = PCA(n_components=self.__n_digits).fit(self.__data)
        self.__execute_pso(title="tSNE PCA PSO Kmeans", gbest_initial=pca.components_)
        tsne_pso = TSNE(n_components=2).fit_transform(self.__pso.gbest)
        tsne = TSNE(n_components=2, init=tsne_pso).fit_transform(self.__data)
        kmeans_tsne = KMeans(n_clusters=self.__n_digits, random_state=0)
        kmeans_tsne.fit_predict(tsne)
        return kmeans_tsne

    def __compare_results(self, kmeans_results, pca_kmeans_results, tsne_kmeans_results, pso_kmeans_results,
                          pso_pca_kmeans_results, tsne_pso_kmeans_results, tsne_pca_pso_kmeans_results):
        print(82 * '_')
        self.__print_metrics("K-means", kmeans_results)
        self.__print_metrics("PCA + K-means", pca_kmeans_results)
        self.__print_metrics("t-SNE + K-means", tsne_kmeans_results)
        self.__print_metrics("PSO K-means", pso_kmeans_results)
        self.__print_metrics("PSO PCA Kmeans", pso_pca_kmeans_results)
        self.__print_metrics("t-SNE PSO Kmeans", tsne_pso_kmeans_results)
        self.__print_metrics("t-SNE PCA PSO Kmeans", tsne_pca_pso_kmeans_results)
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

    def __generate_boxplot(self, kmeans_results, pca_kmeans_results, tsne_kmeans_results, pso_kmeans_results,
                           pso_pca_kmeans_results, tsne_pso_kmeans_results, tsne_pca_pso_kmeans_results):
        data_silhouette = [self.__execute_silhouette(kmeans_results),
                           self.__execute_silhouette(pca_kmeans_results),
                           self.__execute_silhouette(tsne_kmeans_results),
                           self.__execute_silhouette(pso_kmeans_results),
                           self.__execute_silhouette(pso_pca_kmeans_results),
                           self.__execute_silhouette(tsne_pso_kmeans_results),
                           self.__execute_silhouette(tsne_pca_pso_kmeans_results)
                           ]
        fig, ax = plt.subplots()
        ax.set_title('Boxplot Silhouette')
        ax.boxplot(data_silhouette)
        plt.xticks([1, 2, 3, 4, 5, 6, 7], ['Kmeans', 'PCA Kmeans', 't-SNE Kmeans', 'PSO K-means', 'PSO PCA Kmeans',
                                           't-SNE PSO K-means', 't-SNE PCA PSO K-means'])
        plt.savefig('boxplot_silhouette.png')

        plt.close(fig)

        data_v_measure_score = [self.__execute_v_measure_score(kmeans_results),
                                self.__execute_v_measure_score(pca_kmeans_results),
                                self.__execute_v_measure_score(tsne_kmeans_results),
                                self.__execute_v_measure_score(pso_kmeans_results),
                                self.__execute_v_measure_score(pso_pca_kmeans_results),
                                self.__execute_v_measure_score(tsne_pso_kmeans_results),
                                self.__execute_v_measure_score(tsne_pca_pso_kmeans_results)
                                ]
        fig, ax = plt.subplots()
        ax.set_title('Boxplot V Measure Score')
        ax.boxplot(data_v_measure_score)
        plt.xticks([1, 2, 3, 4, 5, 6, 7], ['Kmeans', 'PCA Kmeans', 't-SNE Kmeans', 'PSO K-means', 'PSO PCA Kmeans',
                                           't-SNE PSO K-means', 't-SNE PCA PSO K-means'])
        plt.savefig('boxplot_v_measure_score.png')
        plt.close(fig)

        data_adjusted_rand_score = [self.__execute_adjusted_rand_score(kmeans_results),
                                    self.__execute_adjusted_rand_score(pca_kmeans_results),
                                    self.__execute_adjusted_rand_score(tsne_kmeans_results),
                                    self.__execute_adjusted_rand_score(pso_kmeans_results),
                                    self.__execute_adjusted_rand_score(pso_pca_kmeans_results),
                                    self.__execute_adjusted_rand_score(tsne_pso_kmeans_results),
                                    self.__execute_adjusted_rand_score(tsne_pca_pso_kmeans_results)
                                    ]
        fig, ax = plt.subplots()
        ax.set_title('Boxplot Adjusted Rand_Score')
        ax.boxplot(data_adjusted_rand_score)
        plt.xticks([1, 2, 3, 4, 5, 6, 7], ['Kmeans', 'PCA Kmeans', 't-SNE Kmeans', 'PSO K-means', 'PSO PCA Kmeans',
                                           't-SNE PSO K-means', 't-SNE PCA PSO K-means'])
        plt.savefig('boxplot_adjusted_rand_score.png')
        plt.close(fig)

        data_calinski_harabasz_score = [self.__execute_calinski_harabasz_score(kmeans_results),
                                        self.__execute_calinski_harabasz_score(pca_kmeans_results),
                                        self.__execute_calinski_harabasz_score(tsne_kmeans_results),
                                        self.__execute_calinski_harabasz_score(pso_kmeans_results),
                                        self.__execute_calinski_harabasz_score(tsne_pso_kmeans_results),
                                        self.__execute_calinski_harabasz_score(tsne_pca_pso_kmeans_results)
                                        ]
        fig, ax = plt.subplots()
        ax.set_title('Boxplot Data Calinski Harabasz_Score')
        ax.boxplot(data_calinski_harabasz_score)
        plt.xticks([1, 2, 3, 4, 5, 6, 7], ['Kmeans', 'PCA Kmeans', 't-SNE Kmeans', 'PSO K-means', 'PSO PCA Kmeans',
                                           't-SNE PSO K-means', 't-SNE PCA PSO K-means'])
        plt.savefig('boxplot_calinski_harabasz_score.png')
        plt.close(fig)

    def __execute_silhouette(self, kmeans_array):
        silhouette_array = []
        for kmeans in kmeans_array:
            silhouette_array.append(metrics.silhouette_score(self.__data, kmeans.labels_, metric='euclidean',
                                                             sample_size=self.__sample_size))
        return silhouette_array

    def __execute_v_measure_score(self, kmeans_array):
        v_measure_score_array = []
        for kmeans in kmeans_array:
            v_measure_score_array.append(v_measure_score(self.__labels, kmeans.labels_))
        return v_measure_score_array

    def __execute_adjusted_rand_score(self, kmeans_array):
        adjusted_rand_score_array = []
        for kmeans in kmeans_array:
            adjusted_rand_score_array.append(adjusted_rand_score(self.__labels, kmeans.labels_))
        return adjusted_rand_score_array

    def __execute_calinski_harabasz_score(self, kmeans_array):
        calinski_harabasz_score_array = []
        for kmeans in kmeans_array:
            calinski_harabasz_score_array.append(metrics.calinski_harabasz_score(self.__data, kmeans.labels_))
        return calinski_harabasz_score_array

    @staticmethod
    def __generate_plot(title: str, data):
        fig, ax = plt.subplots()
        ax.set_title(title)
        plt.plot(data)
        filename = title.lower().replace(' ', '_')
        plt.savefig(filename + ".png")

        plt.close(fig)


main = Main()
main.execute()
