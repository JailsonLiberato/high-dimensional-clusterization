from copy import copy
import numpy as np
from deprecated.constants import Constants


class Particle:

    def __init__(self, fitness, position):
        self.__fitness = fitness
        self.__position = position
        self.__pbest = copy(position)
        self.__velocity = np.zeros(Constants.N_DIMENSIONS)

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, fitness):
        self.__fitness = fitness

    @property
    def position(self):
        return self.__position

    @position.setter
    def position(self, position):
        self.__position = position

    @property
    def pbest(self):
        return self.__pbest

    @pbest.setter
    def pbest(self, pbest):
        self.__pbest = pbest

    @property
    def velocity(self):
        return self.__velocity

    @velocity.setter
    def velocity(self, velocity):
        self.__velocity = velocity
