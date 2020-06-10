import math
import unittest
import warnings

import numpy as np

from pyspark.sql import SparkSession

from smart_spark.jobs.optimization import fit_weibull, fit_censored_weibull

from .generate_weibull_data import generate_weibull_data, censor_weibull_data


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
