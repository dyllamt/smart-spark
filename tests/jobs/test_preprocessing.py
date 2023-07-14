import os
import unittest

from pyspark.sql import SparkSession

from smart_spark import data as bb_data
from smart_spark.schema import schema as bb_schema
from smart_spark.jobs.preprocessing import failure_events, partition_by_model


class TestPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        spark = SparkSession.builder.getOrCreate()
        data_dir = os.path.dirname(bb_data.__file__)
        data_files = os.path.join(data_dir, "2015/*.csv.gz")
        cls._data = spark.read.format("csv") \
            .schema(bb_schema) \
            .option("header", "true") \
            .option("enforceSchema", "false") \
            .load(data_files)

    def test_failure_events(self):

        failures = failure_events(self._data)
        self.assertEqual(failures.count(), 1429)

    def test_partition_by_model(self):

        df_by_model = partition_by_model(self._data)
        self.assertEqual(len(df_by_model), 78)  # 78 models in the dataset

        st4000 = df_by_model["ST4000DM000"]
        self.assertEqual(st4000.count(), 6782765)  # 6-million+ records for particular model
