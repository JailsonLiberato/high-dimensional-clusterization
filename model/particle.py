import numpy as np
from copy import copy


class Particle:

    def __init__(self, id, position, fitness, pbest_fitness, n_dimensions):
        self.__fitness = fitness
        self.__pbest_fitness = pbest_fitness
        self.__id = id
        self.__position = position
        self.__pbest = copy(position)
        self.__velocity = np.zeros(n_dimensions)

    @property
    def id(self):
        return self.__id

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, position):
        self.__position = position

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, velocity):
        self.__velocity = velocity

    @property
    def pbest(self):
        return self.__pbest

    @pbest.setter
    def pbest(self, pbest):
        self.__pbest = pbest

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, fitness):
        self.__fitness = fitness

    @property
    def pbest_fitness(self):
        return self.__pbest_fitness

    @pbest_fitness.setter
    def pbest_fitness(self, pbest_fitness):
        self.__pbest_fitness = pbest_fitness
