from sklearn import datasets
from sklearn.metrics import silhouette_score
from utils import Utils
from kmeans import Kmeans
from pso import ParticleSwarmOptimization
import random
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog


class Main:

    kmeans_app = {
        'silhouette': [],
        'sse': [],
        'quantization': [],
    }

    pso_plain = {
        'silhouette': [],
        'sse': [],
        'quantization': [],
    }

    pso_hybrid = {
        'silhouette': [],
        'sse': [],
        'quantization': [],
    }

    def __init__(self, dataset, n_dim):
        self.__X, self.__y = self.__initializate_data_sim(dataset, n_dim)

    @staticmethod
    def __initializate_data_sim(dataset, n_dim):
        columns = random.sample(range(dataset.data.shape[1]), n_dim)
        X = dataset.data[:, columns]
        y = dataset.target
        return X, y

    def execute_kmeans(self):
        print("Executing K-means...")
        for _ in range(2):
            kmeans = Kmeans(n_cluster=3, init_pp=True)
            kmeans.fit(self.__X)
            predicted_kmeans = kmeans.predict(self.__X)
            self.__organize_metrics(self.kmeans_app, kmeans, predicted_kmeans)

    def execute_pso_kmeans(self):
        print("Executing PSO-Kmeans")
        for _ in range(2):
            pso = ParticleSwarmOptimization(
                n_cluster=3, n_particles=10, data=self.__X, hybrid=False, max_iter=2000, print_debug=2000)
            pso.run()
            pso_kmeans = Kmeans(n_cluster=3, init_pp=False)
            pso_kmeans.centroid = pso.gbest_centroids.copy()
            pso_kmeans.fit(self.__X)
            predicted_pso = pso_kmeans.predict(self.__X)
            self.__organize_metrics(self.pso_plain, pso_kmeans, predicted_pso)

    def execute_pso_hybrid(self):
        print("Executing PSO-Hybrid")
        for _ in range(2):
            pso = ParticleSwarmOptimization(
                n_cluster=3, n_particles=10, data=self.__X, hybrid=True, max_iter=2000, print_debug=2000)
            pso.run()
            pso_kmeans = Kmeans(n_cluster=3, init_pp=False)
            pso_kmeans.centroid = pso.gbest_centroids.copy()
            pso_kmeans.fit(self.__X)
            predicted_pso = pso_kmeans.predict(self.__X)
            self.__organize_metrics(self.pso_hybrid, pso_kmeans, predicted_pso)



    def execute_pca_pso_kmeans(self):
        pass

    def execute_tsne_pso_kmeans(self):
        pass

    def __organize_metrics(self, dictionary_data, kmeans: Kmeans, predicted_kmeans):
        silhouette = silhouette_score(self.__X, predicted_kmeans)
        sse = kmeans.sse
        quantization = Utils.quantization_error(centroids=kmeans.centroid, data=self.__X, labels=predicted_kmeans)
        dictionary_data['silhouette'].append(silhouette)
        dictionary_data['sse'].append(sse)
        dictionary_data['quantization'].append(quantization)
        print(dictionary_data)

    def execute_comparison(self):
        print("Executing comparison...\n")
        print(self.kmeans_app['sse'])
        print(self.pso_plain['sse'])

        benchmark = {
            'method': ['K-Means++', 'PSO', 'PSO Hybrid'],
            'sse_mean': [
                np.around(np.mean(self.kmeans_app['sse']), decimals=10),
                np.around(np.mean(self.pso_plain['sse']), decimals=10),
                np.around(np.mean(self.pso_hybrid['sse']), decimals=10),
            ],
            'sse_stdev': [
                np.around(np.std(self.kmeans_app['sse']), decimals=10),
                np.around(np.std(self.pso_plain['sse']), decimals=10),
                np.around(np.std(self.pso_hybrid['sse']), decimals=10),
            ],
            'silhouette_mean': [
                np.around(np.mean(self.kmeans_app['silhouette']), decimals=10),
                np.around(np.mean(self.pso_plain['silhouette']), decimals=10),
                np.around(np.mean(self.pso_hybrid['silhouette']), decimals=10),
            ],
            'silhouette_stdev': [
                np.around(np.std(self.kmeans_app['silhouette']), decimals=10),
                np.around(np.std(self.pso_plain['silhouette']), decimals=10),
                np.around(np.std(self.pso_hybrid['silhouette']), decimals=10),
            ],
            'quantization_mean': [
                np.around(np.mean(self.kmeans_app['quantization']), decimals=10),
                np.around(np.mean(self.pso_plain['quantization']), decimals=10),
                np.around(np.mean(self.pso_hybrid['quantization']), decimals=10),
            ],
            'quantization_stdev': [
                np.around(np.std(self.kmeans_app['quantization']), decimals=10),
                np.around(np.std(self.pso_plain['quantization']), decimals=10),
                np.around(np.std(self.pso_hybrid['quantization']), decimals=10),
            ],
        }
        benchmark_dataframe = pd.DataFrame.from_dict(benchmark)
        benchmark_dataframe.to_csv('benchmark_res.csv', index=False)




dataset = datasets.load_iris()
main = Main(dataset, n_dim=4)
main.execute_kmeans()
main.execute_pso_kmeans()
main.execute_pso_hybrid()
main.execute_pca_pso_kmeans()
main.execute_tsne_pso_kmeans()
main.execute_comparison()
