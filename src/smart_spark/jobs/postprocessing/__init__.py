"""Job steps for postprocessing models for S.M.A.R.T. hard-drive statistics.
"""
from typing import List, Tuple, Union

from pyspark.sql import DataFrame


def bin_counts(df: DataFrame, column: str, bins: Union[float, List[float]]) -> Tuple[List[float], List[float]]:
    """Load bin counts from a single column.

    Parameters
    ----------
    df: DataFrame
        Data collection.
    column: str
        Column name.
    bins: Union[float, List[float]]
        Number of evenly spaced bins or bin endpoints (open right intervals).

    Returns
    -------
    Tuple[List[float], List[float]]
        Bin endpoints (open right intervals) and their bin counts.
    """
    return df.select(column).rdd.flatMap(lambda x: x).histogram(bins)


def numeric_cdf(bins: List[float], counts: List[float]) -> Tuple[List[float], List[float]]:
    """Calculates a cdf from bin counts.

    Parameters
    ----------
    bins: List[float]
        Bin endpoints (open right intervals).
    counts: List[float]
        Bin counts.

    Returns
    -------
    Tuple[List[float], List[float]]
        x and y coordinates describing the cdf.
    """
    bins, counts = bins, counts
    total_samples = sum(counts)
    running_total = 0
    x, y = list(), list()
    for point, count in zip(bins, counts):
        running_total += count
        x.append(point)
        y.append(running_total / total_samples)
    return x, y
