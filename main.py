import pandas as pd
import numpy
import matplotlib.pyplot as plt
from utils import Utils
from pso import ParticleSwarmOptimizedClustering
from particle import quantization_error
from kmeans import Kmeans
from sklearn.metrics import silhouette_score

data = pd.read_csv('seed.txt', sep='\t', header=None)
print(data.head())

x = data.drop([7], axis=1)
x = x.values
x = Utils.normalize(x)
print("\n")
print(x)

print("\n")
print("Kmeans")
kmeans = Kmeans(n_cluster=3, init_pp=False, seed=2018)
kmeans.fit(x)
predicted_kmeans = kmeans.predict(x)
print('Silhouette:', silhouette_score(x, predicted_kmeans))
print('SSE:', kmeans.sse)
print('Quantization:', quantization_error(centroids=kmeans.centroid, data=x, labels=predicted_kmeans))

print("\n")
kmeans2 = Kmeans(n_cluster=3, init_pp=True, seed=2018)
kmeans2.fit(x)
predicted_kmeans2 = kmeans2.predict(x)
print('Silhouette:', silhouette_score(x, predicted_kmeans))
print('SSE:', kmeans2.sse)
print('Quantization:', quantization_error(centroids=kmeans2.centroid, data=x, labels=predicted_kmeans2))

pso = ParticleSwarmOptimizedClustering(
        n_cluster=3, n_particles=10, data=x, hybrid=True, max_iter=2000, print_debug=50)
hist = pso.run()
print(hist)

print("\n")

pso_kmeans = Kmeans(n_cluster=3, init_pp=False, seed=2018)

pso_kmeans.centroid = pso.gbest_centroids.copy()
pso_kmeans.centroid

print(pso_kmeans.centroid)

predicted_pso = pso_kmeans.predict(x)
print('Silhouette:', silhouette_score(x, predicted_pso))
print('SSE:', Utils.calc_sse(centroids=pso.gbest_centroids, data=x, labels=predicted_pso))
print('Quantization:', pso.gbest_score)

kmeanspp = {
    'silhouette': [],
    'sse' : [],
    'quantization' : [],
}
for _ in range(20):
    kmean_rep = Kmeans(n_cluster=3, init_pp=True)
    kmean_rep.fit(x)
    predicted_kmean_rep = kmean_rep.predict(x)
    silhouette = silhouette_score(x, predicted_kmean_rep)
    sse = kmean_rep.sse
    quantization = quantization_error(centroids=kmean_rep.centroid, data=x, labels=predicted_kmean_rep)
    kmeanspp['silhouette'].append(silhouette)
    kmeanspp['sse'].append(sse)
    kmeanspp['quantization'].append(quantization)
print("\n")
print("Kmeans")
print(kmeanspp['silhouette'])
print(kmeanspp['sse'])
print(kmeanspp['quantization'])

pso_plain = {
    'silhouette': [],
    'sse': [],
    'quantization': [],
}
for _ in range(20):
    pso_rep = ParticleSwarmOptimizedClustering(
        n_cluster=3, n_particles=10, data=x, hybrid=False, max_iter=2000, print_debug=2000)
    pso_rep.run()
    pso_kmeans = Kmeans(n_cluster=3, init_pp=False, seed=2018)
    pso_kmeans.centroid = pso_rep.gbest_centroids.copy()
    predicted_pso_rep = pso_kmeans.predict(x)

    silhouette = silhouette_score(x, predicted_pso_rep)
    sse = Utils.calc_sse(centroids=pso_rep.gbest_centroids, data=x, labels=predicted_pso_rep)
    quantization = pso_rep.gbest_score
    pso_plain['silhouette'].append(silhouette)
    pso_plain['sse'].append(sse)
    pso_plain['quantization'].append(quantization)
print("\n")
print("PSO PLAIN")
print(pso_plain['silhouette'])
print(pso_plain['sse'])
print(pso_plain['quantization'])

pso_hybrid = {
    'silhouette': [],
    'sse': [],
    'quantization': [],
}
for _ in range(20):
    pso_rep = ParticleSwarmOptimizedClustering(
        n_cluster=3, n_particles=10, data=x, hybrid=True, max_iter=2000, print_debug=2000)
    pso_rep.run()
    pso_kmeans = Kmeans(n_cluster=3, init_pp=False, seed=2018)
    pso_kmeans.centroid = pso_rep.gbest_centroids.copy()
    predicted_pso_rep = pso_kmeans.predict(x)

    silhouette = silhouette_score(x, predicted_pso_rep)
    sse = Utils.calc_sse(centroids=pso_rep.gbest_centroids, data=x, labels=predicted_pso_rep)
    quantization = pso_rep.gbest_score
    pso_hybrid['silhouette'].append(silhouette)
    pso_hybrid['sse'].append(sse)
    pso_hybrid['quantization'].append(quantization)
print("\n")
print("PSO HYBRID")
print(pso_hybrid['silhouette'])
print(pso_hybrid['sse'])
print(pso_hybrid['quantization'])


benchmark = {
    'method' : ['K-Means++', 'PSO', 'PSO Hybrid'],
    'sse_mean' : [
        numpy.around(numpy.mean(kmeanspp['sse']), decimals=10),
        numpy.around(numpy.mean(pso_plain['sse']), decimals=10),
        numpy.around(numpy.mean(pso_hybrid['sse']), decimals=10),
    ],
    'sse_stdev' : [
        numpy.around(numpy.std(kmeanspp['sse']), decimals=10),
        numpy.around(numpy.std(pso_plain['sse']), decimals=10),
        numpy.around(numpy.std(pso_hybrid['sse']), decimals=10),
    ],
    'silhouette_mean' : [
        numpy.around(numpy.mean(kmeanspp['silhouette']), decimals=10),
        numpy.around(numpy.mean(pso_plain['silhouette']), decimals=10),
        numpy.around(numpy.mean(pso_hybrid['silhouette']), decimals=10),
    ],
    'silhouette_stdev' : [
        numpy.around(numpy.std(kmeanspp['silhouette']), decimals=10),
        numpy.around(numpy.std(pso_plain['silhouette']), decimals=10),
        numpy.around(numpy.std(pso_hybrid['silhouette']), decimals=10),
    ],
    'quantization_mean' : [
        numpy.around(numpy.mean(kmeanspp['quantization']), decimals=10),
        numpy.around(numpy.mean(pso_plain['quantization']), decimals=10),
        numpy.around(numpy.mean(pso_hybrid['quantization']), decimals=10),
    ],
    'quantization_stdev' : [
        numpy.around(numpy.std(kmeanspp['quantization']), decimals=10),
        numpy.around(numpy.std(pso_plain['quantization']), decimals=10),
        numpy.around(numpy.std(pso_hybrid['quantization']), decimals=10),
    ],
}

print("\n")
print(benchmark)

print("\n")
benchmark_df = pd.DataFrame.from_dict(benchmark)
print(benchmark_df)