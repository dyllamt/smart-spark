import unittest
import math

from pyspark.sql import SparkSession

from smart_spark.entity.model.weibull_model import WeibullModel


class TestWeibullModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._model = WeibullModel(  # sets-up exponential distribution to test on
            shape=1.,
            scale=1.
        )

        sc = SparkSession.builder.getOrCreate().sparkContext
        cls._data = sc.parallelize(
            [0., 0., 0.]
        )

    def test_pdf(self):

        self.assertEqual(self._model.pdf(0), 1)
        self.assertAlmostEqual(self._model.pdf(math.e), 0.0659, 3)

    def test_log_likelihood(self):

        self.assertEqual(self._model.log_likelihood(self._data), 0)
