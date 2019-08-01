from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from business.pso import ParticleSwarmOptimization
from util.plot_util import PlotUtil
from util.constants import Constants


class MainFunctions:
    """Main functions"""

    KMEANS = "kmeans"
    PCA_KMEANS = "pca_kmeans"
    TSNE_KMEANS = "tsne_kmeans"
    PSO_KMEANS = "pso_kmeans"
    PSO_PCA_KMEANS = "pso_pca_kmeans"
    TSNE_PSO_KMEANS = "tsne_pso_kmeans"
    TSNE_PCA_PSO_KMEANS = "tsne_pca_pso_kmeans"

    @staticmethod
    def execute_kmeans(n_clusters, data):
        print("Executing K-means...")
        kmeans = KMeans(n_clusters=n_clusters, init='random')
        kmeans.fit(data)
        return kmeans

    @staticmethod
    def execute_pca_kmeans(n_clusters, data):
        print("Executing PCA K-means...")
        pca = PCA(n_components=n_clusters).fit(data)
        kmeans = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
        kmeans.fit(X=data)
        return kmeans

    @staticmethod
    def execute_tsne_kmeans(n_clusters, data):
        print("Executing t-SNE K-means...")
        tsne = TSNE(n_components=2).fit_transform(X=data)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit_predict(X=tsne)
        return kmeans

    @staticmethod
    def execute_pso_kmeans(n_clusters, data):
        print("Executing PSO K-means...")
        pso = MainFunctions.execute_pso(title="PSO Kmeans", data=data)
        kmeans = KMeans(init=pso.gbest, n_clusters=n_clusters, n_init=1)
        kmeans.fit(data)
        return kmeans

    @staticmethod
    def execute_pso_pca_kmeans(n_clusters, data):
        print("Executing PSO PCA K-means...")
        pca = PCA(n_components=n_clusters).fit(X=data)
        pso = MainFunctions.execute_pso(title="PSO PCA Kmeans", data=data, gbest_initial=pca.components_)
        kmeans = KMeans(init=pso.gbest, n_clusters=n_clusters, n_init=1)
        kmeans.fit(data)
        return kmeans

    @staticmethod
    def execute_pso(title, data, gbest_initial=None):
        pso = ParticleSwarmOptimization(n_dimensions=Constants.N_DIMENSIONS, data=data,
                                        gbest_initial=gbest_initial)
        pso.optimize()
        PlotUtil.generate_plot(title + " Convergency", pso.gbest_fitness_array)
        return pso

    # Problem: Discovering about the integration.
    @staticmethod
    def execute_tsne_pso_kmeans(n_clusters, data):
        print("Executing t-SNE PSO K-means...")
        pso = MainFunctions.execute_pso(title="tSNE PSO Kmeans", data=data)
        tsne = TSNE(n_components=2).fit_transform(X=data)
        kmeans_tsne = KMeans(n_clusters=n_clusters)
        kmeans_tsne.fit_predict(tsne)
        return kmeans_tsne

    # Problem: Discovering about the integration.
    @staticmethod
    def execute_tsne_pca_pso_kmeans(n_clusters, data):
        print("Executing t-SNE PCA PSO K-means...")
        pca = PCA(n_components=n_clusters).fit(X=data)
        pso = MainFunctions.execute_pso(title="tSNE PCA PSO Kmeans", data=data, gbest_initial=pca.components_)
        tsne = TSNE(n_components=2).fit_transform(data)
        kmeans_tsne = KMeans(n_clusters=n_clusters)
        kmeans_tsne.fit_predict(tsne)
        return kmeans_tsne
