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
        spark = SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        cls._data = sc.parallelize(
            [0., 0., 0.]
        )
        cls._censored_data_with_events = spark.createDataFrame(
            [
                (0., 1.),  # data where events occurred
                (0., 1.),
                (0., 1.),
            ],
            ['data', 'censor']  # column labels
        )
        cls._censored_data_without_events = spark.createDataFrame(
            [
                (0., 0.),  # data where events did not occur
                (0., 0.),
                (0., 0.),
            ],
            ['data', 'censor']  # column labels
        )

    def test_pdf(self):

        self.assertEqual(self._model.pdf(0), 1)
        self.assertAlmostEqual(self._model.pdf(math.e), 0.0659, 3)

    def test_cdf(self):

        self.assertEqual(self._model.cdf(0), 0)
        self.assertAlmostEqual(self._model.cdf(math.e), 0.9340, 3)
        self.assertAlmostEqual(self._model.cdf(1000000000.), 1., 3)

    def test_log_likelihood(self):

        self.assertEqual(self._model.log_likelihood(self._data), 0)

    def test_right_censored_log_likelihood(self):

        self.assertEqual(self._model.right_censored_log_likelihood(self._censored_data_with_events), 0)
        self.assertEqual(self._model.right_censored_log_likelihood(self._censored_data_without_events), 0)
