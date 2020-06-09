from typing import List, Tuple

import math
import random
import unittest
import warnings

import numpy as np

from pyspark.sql import SparkSession

from smart_spark.jobs.optimization import fit_weibull, fit_censored_weibull


def inverted_weibull_cdf(x: float, shape: float, scale: float) -> float:
    return scale * ((- math.log(1 - x))**(1. / shape))


def generate_weibull_data(points: int, shape: float, scale: float, seed=None) -> List[float]:
    random.seed(a=seed)
    data = [random.random() for i in range(points)]
    data = [inverted_weibull_cdf(i, shape, scale) for i in data]
    return data


def censor_weibull_data(data: List[float], event_modifier: float, seed=None) -> List[Tuple[float, float]]:
    random.seed(a=seed)
    max_var = max(data)
    observations = [random.random() * max_var * event_modifier for i in data]
    censor = [1. if o > d else 0. for o, d in zip(observations, data)]
    data = [min(d, o) for d, o in zip(data, observations)]
    return list(zip(data, censor))


class TestOptimization(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        warnings.simplefilter("ignore", ResourceWarning)
        cls._spark = SparkSession.builder.getOrCreate()
        cls._sc = cls._spark.sparkContext

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

    def test_fit_censored_weibull(self):  # fraction of censoring has no material impact on fit

        dist_params = (2., 2.)  # 25% censoring
        data = self._spark.createDataFrame(
            censor_weibull_data(
                generate_weibull_data(100000, *dist_params, seed=13457847445),
                event_modifier=1.,
                seed=1432764
            ),
            ["data", "censor"])
        output = fit_censored_weibull(data, method="Nelder-Mead")
        fit_params = (output.parameters["shape"], output.parameters["scale"])
        for dist, fit in zip(dist_params, fit_params):
            self.assertTrue(math.isclose(dist, fit, rel_tol=0.05))

        dist_params = (2., 2.)  # 50% censoring
        data = self._spark.createDataFrame(
            censor_weibull_data(
                generate_weibull_data(100000, *dist_params, seed=13457847445),
                event_modifier=0.5,
                seed=1432764
            ),
            ["data", "censor"])
        output = fit_censored_weibull(data, method="Nelder-Mead")
        fit_params = (output.parameters["shape"], output.parameters["scale"])
        for dist, fit in zip(dist_params, fit_params):
            self.assertTrue(math.isclose(dist, fit, rel_tol=0.05))

        dist_params = (2., 2.)  # 90% censoring
        data = self._spark.createDataFrame(
            censor_weibull_data(
                generate_weibull_data(100000, *dist_params, seed=13457847445),
                event_modifier=.17,
                seed=1432764
            ),
            ["data", "censor"])
        output = fit_censored_weibull(data, method="Nelder-Mead")
        fit_params = (output.parameters["shape"], output.parameters["scale"])
        for dist, fit in zip(dist_params, fit_params):
            self.assertTrue(math.isclose(dist, fit, rel_tol=0.05))
