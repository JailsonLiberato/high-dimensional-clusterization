import numpy as np
from particle import Particle


class ParticleSwarmOptimization:

    def __init__(self,
                 n_cluster: int, n_particles: int, data: np.ndarray, hybrid: bool = True, max_iter: int = 100,
                 print_debug: int = 10):
        self.__n_cluster = n_cluster
        self.__n_particles = n_particles
        self.__data = data
        self.__max_iter = max_iter
        self.__particles = []
        self.__hybrid = hybrid
        self.__print_debug = print_debug
        self.__gbest_score = np.inf
        self.__gbest_centroids = None
        self.__gbest_sse = np.inf
        self.__init_particles()

    def __init_particles(self):
        for i in range(self.__n_particles):
            particle = None
            if i == 0 and self.__hybrid:
                particle = Particle(self.__n_cluster, self.__data, use_kmeans=True)
            else:
                particle = Particle(self.__n_cluster, self.__data, use_kmeans=False)
            if particle.best_score < self.__gbest_score:
                self.__gbest_centroids = particle.centroids.copy()
                self.__gbest_score = particle.best_score
            self.__particles.append(particle)
            self.__gbest_sse = min(particle.best_sse, self.__gbest_sse)

    def run(self):
        print('Initial global best score', self.__gbest_score)
        history = []
        for i in range(self.__max_iter):
            for particle in self.__particles:
                particle.update(self.__gbest_centroids, self.__data)
            for particle in self.__particles:
                if particle.best_score < self.__gbest_score:
                    self.__gbest_centroids = particle.centroids.copy()
                    self.__gbest_score = particle.best_score
            history.append(self.__gbest_score)
            if i % self.__print_debug == 0:
                print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(
                    i + 1, self.__max_iter, self.__gbest_score))
        print('Finish with gbest score {:.18f}'.format(self.__gbest_score))
        return history
