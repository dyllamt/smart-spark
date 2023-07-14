import math
import unittest
import warnings

from pyspark.sql import SparkSession

from smart_spark.jobs.postprocessing import bin_counts, numeric_cdf
from smart_spark.weibull import Weibull

from .generate_weibull_data import generate_weibull_data, censor_weibull_data


class TestPostprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        warnings.simplefilter("ignore", ResourceWarning)
        cls._spark = SparkSession.builder.getOrCreate()
        cls._sc = cls._spark.sparkContext

        dist_params = (2., 2.)  # 25% censoring
        cls._data = cls._spark.createDataFrame(
            censor_weibull_data(
                generate_weibull_data(100000, *dist_params, seed=13457845),
                event_modifier=10000000.,  # eliminate censoring
                seed=14364
            ),
            ["data", "censor"]
        )

    def test_bin_counts(self):

        expected_counts = [9860, 23908, 26496, 20260, 11726, 5141, 1903, 560, 120, 26]
        edges, counts = bin_counts(self._data, column="data", bins=10)
        for c, e in zip(counts, expected_counts):
            self.assertEqual(c, e, "expected count number ({}) got ({})".format(e, c))

    def test_numeric_cdf(self):  # only testable when not using censored data

        edges, counts = bin_counts(self._data, column="data", bins=100)
        x, y = numeric_cdf(edges, counts)

        model = Weibull(2., 2.)
        s = [model.cdf(i) for i in x]

        for expected, actual in zip(s, y):
            self.assertTrue(math.isclose(actual, expected, abs_tol=0.05))
