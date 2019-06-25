import numpy


class Utils:

    @staticmethod
    def calc_sse(centroids: numpy.ndarray, labels: numpy.ndarray, data: numpy.ndarray):
        distances = 0
        for i, c in enumerate(centroids):
            idx = numpy.where(labels == i)
            dist = numpy.sum((data[idx] - c) ** 2)
            distances += dist
        return distances



    @staticmethod
    def normalize(x: numpy.ndarray):
        return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    @staticmethod
    def standardize(x: numpy.ndarray):
        return (x - x.mean(axis=0)) / numpy.std(x, axis=0)