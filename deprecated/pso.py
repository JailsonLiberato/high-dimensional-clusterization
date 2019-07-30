from copy import copy
import numpy as np
from deprecated.particle import Particle


class ParticleSwarmOptimization:

    def __init__(self, max_iterations, fitness_function):
        self.__max_iterations = max_iterations
        self.__fitness_function = fitness_function

    def execute(self):
        count_iterations: int = 0
        while count_iterations < self.__max_iterations:
            self.__calculate_fitness()
            self.__update_gbest()
            self.__calculate_velocity()
            self.__update_position()
            self.__update_bound_adjustament()
            count_iterations += 1

    def initialize_particles(self):
        particles = []
        for i in range(Constants.N_PARTICLES):
            position = self.generate_initial_position(fitness_function)
            particle = Particle(i + 1, position, fitness_function.run(position))
            particles.append(particle)
        return particles

    @staticmethod
    def generate_initial_position(fitness_function: FitnessFunction):
        min_value = fitness_function.min_initialization
        max_value = fitness_function.max_initialization
        return np.random.uniform(min_value, max_value, size=(1, Constants.N_DIMENSIONS))[0]

    def __calculate_fitness(self):
        for particle in self.__particles:
            if self.__fitness_function(particle.position) > self.__fitness_function(particle.pbest):
                particle.pbest = copy(particle.position)
                particle.fitness = self.__fitness_function(particle.position)

    def __update_gbest(self):
        for particle in self.__particles:
            if self.__fitness_function(particle.pbest) > self.__fitness_function(self.__gbest):
                self.__gbest = particle.pbest