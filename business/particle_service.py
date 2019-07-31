from model.particle import Particle
import numpy as np
from util.constants import Constants
from copy import copy


class ParticleService:

    def initialize_particles(self, min_bound, max_bound, n_dimensions, fitness_function, data):
        particles = []
        for i in range(Constants.N_PARTICLES):
            position = self.__generate_initial_position(min_bound, max_bound, n_dimensions)
            fitness = fitness_function.run(position, data)
            particle = Particle(i + 1, position, fitness, 0, n_dimensions)
            particles.append(particle)
        return particles

    @staticmethod
    def __generate_initial_position(min_bound, max_bound, n_dimensions):
        array = np.random.uniform(min_bound, max_bound, size=(Constants.N_CLUSTERS, n_dimensions))
        return array
