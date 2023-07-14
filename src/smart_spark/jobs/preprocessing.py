"""Job steps for preprocessing S.M.A.R.T. hard-drive statistics.
"""
from typing import Dict

from pyspark.sql import DataFrame


def failure_events(df: DataFrame) -> DataFrame:
    """Returns a subset of the data, where failures occurred.

    Parameters
    ----------
    df: DataFrame
        Dataset on hard-drive S.M.A.R.T. statistics.

    Returns
    -------
    DataFrame
        Dataset rows where a failure has occurred.
    """
    return df.filter(df.failure == 1)


def partition_by_model(df: DataFrame) -> Dict[str, DataFrame]:
    """Returns multiple partitions of the dataset by hard-drive model.

    Parameters
    ----------
    df: DataFrame
        Dataset on hard-drive S.M.A.R.T. statistics.

    Returns
    -------
    Dict[DataFrame]
        Datasets split on hard-drive model.
    """
    hd_models = df.select("model").distinct().collect()  # action
    return {m.model: df.filter(df.model == m.model) for m in hd_models}
