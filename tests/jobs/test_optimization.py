from typing import List

import math
import random
import unittest
import warnings

import numpy as np

from pyspark.sql import SparkSession

from smart_spark.jobs.optimization import fit_weibull


def inverted_weibull_cdf(x: float, shape: float, scale: float) -> float:
    return scale * ((- math.log(1 - x))**(1. / shape))


def generate_weibull_data(points: int, shape: float, scale: float, seed=None) -> List[float]:
    random.seed(a=seed)
    data = [random.random() for i in range(points)]
    data = [inverted_weibull_cdf(i, shape, scale) for i in data]
    return data


class TestOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        warnings.simplefilter("ignore", ResourceWarning)
        cls._sc = SparkSession.builder.getOrCreate().sparkContext

    def test_fit_weibull(self):

        dist_params = (2., 2.)
        data = self._sc.parallelize(generate_weibull_data(100000, *dist_params, seed=13457847445))
        output = fit_weibull(data, method="Nelder-Mead")
        fit_params = (output.parameters["shape"], output.parameters["scale"])
        for dist, fit in zip(dist_params, fit_params):
            self.assertTrue(math.isclose(dist, fit, rel_tol=0.05))

        dist_params = (0.6, 2.)
        data = self._sc.parallelize(generate_weibull_data(100000, *dist_params, seed=13457847445))
        output = fit_weibull(data, x0=np.array([1., 5.]), method="Nelder-Mead")
        fit_params = (output.parameters["shape"], output.parameters["scale"])
        for dist, fit in zip(dist_params, fit_params):
            self.assertTrue(math.isclose(dist, fit, rel_tol=0.05))

        dist_params = (2., 150000000.)
        data = self._sc.parallelize(generate_weibull_data(100000, *dist_params, seed=13457847445))
        output = fit_weibull(data, x0=np.array([1.5, 300000000.]), method="Nelder-Mead")
        fit_params = (output.parameters["shape"], output.parameters["scale"])
        for dist, fit in zip(dist_params, fit_params):
            self.assertTrue(math.isclose(dist, fit, rel_tol=0.05))
