from sklearn import datasets
from sklearn.metrics import silhouette_score
from utils import Utils
from kmeans import Kmeans
from pso import ParticleSwarmOptimization
import random
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


class Main:

    kmeans_app = {
        'silhouette': [],
        'sse': [],
        'quantization': []
    }

    pso_plain = {
        'silhouette': [],
        'sse': [],
        'quantization': []
    }

    pso_hybrid = {
        'silhouette': [],
        'sse': [],
        'quantization': []
    }

    tsne_kmeans = {
        'silhouette': [],
        'sse': [],
        'quantization': []
    }

    tsne_pso_kmeans = {
        'silhouette': [],
        'sse': [],
        'quantization': []
    }

    def __init__(self, dataset, n_dim):
        self.__dataset = dataset
        self.__n_dim = n_dim
        self.__x, self.__y = self.__initializate_data_sim()

    def __initializate_data_sim(self):
        columns = random.sample(range(self.__dataset.data.shape[1]), self.__n_dim)
        x = dataset.data[:, columns]
        y = dataset.target
        return x, y

    def execute_kmeans(self):
        print("Executing K-means...")
        for _ in range(20):
            kmeans = Kmeans(n_cluster=3, init_pp=True)
            kmeans.fit(self.__x)
            predicted_kmeans = kmeans.predict(self.__x)
            self.__organize_metrics(self.kmeans_app, kmeans, predicted_kmeans)

    def execute_pso_kmeans(self):
        print("Executing PSO-Kmeans")
        for _ in range(20):
            pso = ParticleSwarmOptimization(
                n_cluster=3, n_particles=10, data=self.__x, hybrid=False, max_iter=2000, print_debug=2000)
            pso.run()
            pso_kmeans = Kmeans(n_cluster=3, init_pp=False)
            pso_kmeans.centroid = pso.gbest_centroids.copy()
            pso_kmeans.fit(self.__x)
            predicted_pso = pso_kmeans.predict(self.__x)
            self.__organize_metrics(self.pso_plain, pso_kmeans, predicted_pso)

    def execute_pso_hybrid(self):
        print("Executing PSO-Hybrid")
        for _ in range(20):
            pso = ParticleSwarmOptimization(
                n_cluster=3, n_particles=10, data=self.__x, hybrid=True, max_iter=2000, print_debug=2000)
            pso.run()
            pso_kmeans = Kmeans(n_cluster=3, init_pp=False)
            pso_kmeans.centroid = pso.gbest_centroids.copy()
            pso_kmeans.fit(self.__x)
            predicted_pso = pso_kmeans.predict(self.__x)
            self.__organize_metrics(self.pso_hybrid, pso_kmeans, predicted_pso)



    def execute_pca_pso_kmeans(self):
        pass

    def execute_tsne_kmeans(self):
        print("Executing t-SNE Kmeans")
        self.__x, self.__y = self.__initializate_data_sim()
        self.__x = TSNE(n_components=2).fit_transform(self.__x)
        for _ in range(20):
            kmeans = Kmeans(n_cluster=3, init_pp=True)
            kmeans.fit(self.__x)
            predicted_kmeans = kmeans.predict(self.__x)
            self.__organize_metrics(self.tsne_kmeans, kmeans, predicted_kmeans)

    def execute_tsne_pso_kmeans(self):
        print("Executing t-SNE PSO Kmeans")
        self.__x, self.__y = self.__initializate_data_sim()
        self.__x = TSNE(n_components=2).fit_transform(self.__x)
        for _ in range(20):
            pso = ParticleSwarmOptimization(
                n_cluster=3, n_particles=10, data=self.__x, hybrid=False, max_iter=2000, print_debug=2000)
            pso.run()
            pso_kmeans = Kmeans(n_cluster=3, init_pp=False)
            pso_kmeans.centroid = pso.gbest_centroids.copy()
            pso_kmeans.fit(self.__x)
            predicted_pso = pso_kmeans.predict(self.__x)
            self.__organize_metrics(self.tsne_pso_kmeans, pso_kmeans, predicted_pso)

    def __organize_metrics(self, dictionary_data, kmeans: Kmeans, predicted_kmeans):
        silhouette = silhouette_score(self.__x, predicted_kmeans)
        sse = kmeans.sse
        quantization = Utils.quantization_error(centroids=kmeans.centroid, data=self.__x, labels=predicted_kmeans)
        dictionary_data['silhouette'].append(silhouette)
        dictionary_data['sse'].append(sse)
        dictionary_data['quantization'].append(quantization)
        print(dictionary_data)

    def execute_comparison2(self, filename):
        benchmark = {
            'method': ['K-means', 't-SNE Kmeans', 't-SNE PSO Kmeans'],
            'sse_mean': [
                np.around(np.mean(self.kmeans_app['sse']), decimals=10),
                np.around(np.mean(self.tsne_kmeans['sse']), decimals=10),
                np.around(np.mean(self.tsne_pso_kmeans['sse']), decimals=10)
            ],
            'sse_stdev': [
                np.around(np.std(self.kmeans_app['sse']), decimals=10),
                np.around(np.std(self.tsne_kmeans['sse']), decimals=10),
                np.around(np.std(self.tsne_pso_kmeans['sse']), decimals=10)
            ],
            'silhouette_mean': [
                np.around(np.mean(self.kmeans_app['silhouette']), decimals=10),
                np.around(np.mean(self.tsne_kmeans['silhouette']), decimals=10),
                np.around(np.mean(self.tsne_pso_kmeans['silhouette']), decimals=10)
            ],
            'silhouette_stdev': [
                np.around(np.std(self.kmeans_app['silhouette']), decimals=10),
                np.around(np.std(self.tsne_kmeans['silhouette']), decimals=10),
                np.around(np.std(self.tsne_pso_kmeans['silhouette']), decimals=10)
            ],
            'quantization_mean': [
                np.around(np.mean(self.kmeans_app['quantization']), decimals=10),
                np.around(np.mean(self.tsne_kmeans['quantization']), decimals=10),
                np.around(np.mean(self.tsne_pso_kmeans['quantization']), decimals=10)
            ],
            'quantization_stdev': [
                np.around(np.std(self.kmeans_app['quantization']), decimals=10),
                np.around(np.std(self.tsne_kmeans['quantization']), decimals=10),
                np.around(np.std(self.tsne_pso_kmeans['quantization']), decimals=10)
            ],
        }

        benchmark_dataframe = pd.DataFrame.from_dict(benchmark)
        benchmark_dataframe.to_csv(filename + '.csv', index=False)

    def execute_comparison(self, filename):
        print("Executing comparison...\n")
        print(self.kmeans_app['sse'])
        print(self.pso_plain['sse'])

        benchmark = {
            'method': ['K-Means++', 'PSO', 'PSO Hybrid', 't-SNE Kmeans','t-SNE PSO Kmeans'],
            'sse_mean': [
                np.around(np.mean(self.kmeans_app['sse']), decimals=10),
                np.around(np.mean(self.pso_plain['sse']), decimals=10),
                np.around(np.mean(self.pso_hybrid['sse']), decimals=10),
                np.around(np.mean(self.tsne_kmeans['sse']), decimals=10),
                np.around(np.mean(self.tsne_pso_kmeans['sse']), decimals=10)
            ],
            'sse_stdev': [
                np.around(np.std(self.kmeans_app['sse']), decimals=10),
                np.around(np.std(self.pso_plain['sse']), decimals=10),
                np.around(np.std(self.pso_hybrid['sse']), decimals=10),
                np.around(np.std(self.tsne_kmeans['sse']), decimals=10),
                np.around(np.std(self.tsne_pso_kmeans['sse']), decimals=10)
            ],
            'silhouette_mean': [
                np.around(np.mean(self.kmeans_app['silhouette']), decimals=10),
                np.around(np.mean(self.pso_plain['silhouette']), decimals=10),
                np.around(np.mean(self.pso_hybrid['silhouette']), decimals=10),
                np.around(np.mean(self.tsne_kmeans['silhouette']), decimals=10),
                np.around(np.mean(self.tsne_pso_kmeans['silhouette']), decimals=10)
            ],
            'silhouette_stdev': [
                np.around(np.std(self.kmeans_app['silhouette']), decimals=10),
                np.around(np.std(self.pso_plain['silhouette']), decimals=10),
                np.around(np.std(self.pso_hybrid['silhouette']), decimals=10),
                np.around(np.std(self.tsne_kmeans['silhouette']), decimals=10),
                np.around(np.std(self.tsne_pso_kmeans['silhouette']), decimals=10)
            ],
            'quantization_mean': [
                np.around(np.mean(self.kmeans_app['quantization']), decimals=10),
                np.around(np.mean(self.pso_plain['quantization']), decimals=10),
                np.around(np.mean(self.pso_hybrid['quantization']), decimals=10),
                np.around(np.mean(self.tsne_kmeans['quantization']), decimals=10),
                np.around(np.mean(self.tsne_pso_kmeans['quantization']), decimals=10)
            ],
            'quantization_stdev': [
                np.around(np.std(self.kmeans_app['quantization']), decimals=10),
                np.around(np.std(self.pso_plain['quantization']), decimals=10),
                np.around(np.std(self.pso_hybrid['quantization']), decimals=10),
                np.around(np.std(self.tsne_kmeans['quantization']), decimals=10),
                np.around(np.std(self.tsne_pso_kmeans['quantization']), decimals=10)
            ],
        }
        benchmark_dataframe = pd.DataFrame.from_dict(benchmark)
        benchmark_dataframe.to_csv(filename + '.csv', index=False)


dataset = datasets.load_wine()
main = Main(dataset, n_dim=13)
main.execute_kmeans()
main.execute_pso_kmeans()
main.execute_pso_hybrid()
main.execute_pca_pso_kmeans()
main.execute_tsne_kmeans()
main.execute_tsne_pso_kmeans()
main.execute_comparison("arquivo")

main2 = Main(dataset, n_dim=13)
main2.execute_kmeans()
main2.execute_tsne_kmeans()
main2.execute_tsne_pso_kmeans()
main2.execute_comparison2("arquivo2")