from sklearn import datasets
from sklearn.metrics import silhouette_score
from utils import Utils
from kmeans import Kmeans
import random


class Main:

    def __init__(self, dataset, n_dim):
        self.__X, self.__y = self.__initializate_data_sim(dataset, n_dim)

    @staticmethod
    def __initializate_data_sim(dataset, n_dim):
        columns = random.sample(range(dataset.data.shape[1]), n_dim)
        X = dataset.data[:, columns]
        y = dataset.target
        return X, y

    def execute_kmeans(self):
        kmeans = Kmeans(n_cluster=3, init_pp=False)
        kmeans.fit(self.__X)
        predicted_kmeans = kmeans.predict(self.__X)
        self.__print_metrics("K-means++", kmeans, predicted_kmeans)

    def execute_pso_kmeans(self):
        pass

    def execute_pca_pso_kmeans(self):
        pass

    def execute_tsne_pso_kmeans(self):
        pass

    def __print_metrics(self, title, kmeans: Kmeans, predicted_kmeans):
        print("[", title, "]\n")
        print('Silhouette:', silhouette_score(self.__X, predicted_kmeans))
        print('SSE:', kmeans.sse)
        print('Quantization:', Utils.quantization_error(centroids=kmeans.centroid, data=self.__X,
                                                        labels=predicted_kmeans))


dataset = datasets.load_iris()
main = Main(dataset, n_dim=2)
main.execute_kmeans()
main.execute_pso_kmeans()
main.execute_pca_pso_kmeans()
main.execute_tsne_pso_kmeans()
