from business.particle_service import ParticleService
from util.constants import Constants
from business.fitness_function import FitnessFunction
from copy import copy
import numpy as np
import random


class ParticleSwarmOptimization:

    def __init__(self, n_dimensions, data):
        self.__fitness_function = FitnessFunction()
        self.__min_bounds = np.amin(data)
        self.__max_bounds = np.amax(data)
        particle_service = ParticleService()
        self.__particles = particle_service.initialize_particles(self.__min_bounds, self.__max_bounds,
                                                                  n_dimensions, self.__fitness_function, data)
        self.__gbest = copy(self.__particles[0].pbest)
        self.__data = data

    def optimize(self):
        count_iterations: int = 0
        while count_iterations < Constants.N_MAX_ITERATIONS:
            for particle in self.__particles:
                self.__calculate_fitness(particle)
                self.__update_gbest(particle)
                inertia = self.__generate_inertia(count_iterations)
                self.__calculate_velocity(inertia, particle)
                self.__update_position(particle)
                self.__update_bound_adjustament(particle)
            count_iterations += 1
            print("PSO[", count_iterations, "]: gBest -> ", self.__fitness_function.run(self.__gbest, self.__data))

    def __calculate_fitness(self, particle):
        if not np.array_equal(particle.position, particle.pbest) and self.__fitness_function.run(particle.position, self.__data) > \
                self.__fitness_function.run(particle.pbest, self.__data):
            particle.pbest = copy(particle.position)
            particle.fitness = self.__fitness_function.run(particle.pbest, self.__data)

    def __update_gbest(self, particle):
        if not np.array_equal(self.__gbest, particle.pbest) and (self.__fitness_function.run(particle.pbest,
                                                                                             self.__data) >
                                                                 self.__fitness_function.run(self.__gbest,
                                                                                             self.__data)) and \
                self.__is_limit_exceeded(particle.pbest):
            self.__gbest = copy(particle.pbest)

    def __is_limit_exceeded(self, pbest):
        return np.any(pbest >= self.__min_bounds) and np.any(pbest <= self.__max_bounds)

    def __calculate_velocity(self, inertia: float, particle):
            r1 = random.uniform(0, 1)
            r2 = random.uniform(0, 1)
            particle.velocity = (inertia * particle.velocity) + Constants.COEFFICIENT1 * r1 * \
                                (particle.pbest - particle.position) + Constants.COEFFICIENT2 * r2 \
                                * (self.__gbest - particle.position)

    def __update_position(self, particle):
        particle.position += particle.velocity

    def __update_bound_adjustament(self, particle):
        min_array = [self.__min_bounds]
        max_array = [self.__max_bounds]
        np.putmask(particle.position, particle.position > max_array, self.__max_bounds)
        np.putmask(particle.position, particle.position < min_array, self.__min_bounds)

    @staticmethod
    def __generate_inertia(count_iterations):
        return Constants.INERTIA_MAX - count_iterations * (Constants.INERTIA_MAX - Constants.INERTIA_MIN) / \
               Constants.N_MAX_ITERATIONS

    @property
    def gbest(self):
        return self.__gbest
