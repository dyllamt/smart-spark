import os
import unittest

from pyspark.sql import SparkSession

from smart_spark.schema import schema as bb_schema


class TestBackBlazeDataSchema(unittest.TestCase):
    _data = None  # type: ignore

    @classmethod
    def setUpClass(cls) -> None:

        spark = SparkSession.builder.getOrCreate()
        data_files = os.path.dirname("../../test_files/2015/*.csv.gz")
        cls._data = spark.read.format("csv") \
            .schema(bb_schema) \
            .option("header", "true") \
            .option("enforceSchema", "false") \
            .load(data_files)

    def test_data_schema(self):

        self.assertEqual(self._data.schema, bb_schema)  # asserts that we have correctly documented the schema

    def test_data_count(self):

        self.assertEqual(17509251, self._data.count())  # asserts consistent number of data records
