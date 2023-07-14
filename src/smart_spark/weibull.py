"""Implementation of weibull states.
"""
import math
from abc import ABC, abstractmethod
from typing import Dict

from pyspark.rdd import RDD
from pyspark.sql import DataFrame, Row


class BaseDistribution(ABC):
    """Parametric model describing a system.

    Parameters
    ----------
    parameters: Dict[str, float]
        The named parameters of this model.
    """

    def __init__(self, parameters: Dict[str, float]):
        self.parameters = parameters

    @abstractmethod
    def log_likelihood(self, rdd: RDD) -> float:
        """Returns the log-likelihood that this model generated the given data distribution.

        Parameters
        ----------
        rdd: RDD
            Elements are assumed to be floats.

        Returns
        -------
        float
            The log-likelihood that this model generated the given data distribution.
        """
        return 0.


class Weibull(BaseDistribution):
    """Two-parameter Weibull distribution of a single variable.

    See the standard parameterization at https://en.wikipedia.org/wiki/Weibull_distribution.

    Parameters
    ----------
    shape: float
        The shape parameter (k).
    scale: float
        The scale parameter (lambda).

    Attributes
    ----------
    parameters: Dict[str, float]
        The named-parameters for this model.
    """

    def __init__(self, shape: float, scale: float):
        parameters = {"shape": shape, "scale": scale}
        super().__init__(parameters)

    def pdf(self, x: float) -> float:
        """The probability density function (of failure) over the single variable.

        Parameters
        ----------
        x: float
            The single variable.

        Returns
        -------
        float
            The probability density of failure at x.
        """
        shape = self.parameters["shape"]
        scale = self.parameters["scale"]
        return (shape / scale) * ((x / scale)**(shape - 1.)) * math.exp(-((x / scale)**shape))

    def cdf(self, x: float) -> float:
        """The cumulative distribution function (of failure) over a single variable.

        Parameters
        ----------
        x: float
            The single variable.

        Returns
        -------
        float
            The cumulative probability of failure up to x.
        """
        shape = self.parameters["shape"]
        scale = self.parameters["scale"]
        return 1. - math.exp(-((x / scale)**shape))

    def log_likelihood(self, rdd: RDD) -> float:
        """Returns the log-likelihood that this model generated the given data distribution.

        Parameters
        ----------
        rdd: RDD
            Elements are assumed to be floats.

        Returns
        -------
        float
            The log-likelihood that this model generated the given data distribution.
        """
        return rdd.map(lambda x: math.log(self.pdf(x))).sum()  # action

    def right_censored_log_likelihood(self, df: DataFrame, censor_col: str = "censor", data_col: str = "data") -> float:
        """Returns the log-likelihood that this model generated the given data distribution (right-censored).

        Parameters
        ----------
        df: DataFrame
            Data collection containing data and a censor.
        censor_col: str
            Column name for the censor (1 if the event occurred and 0 if censored).
        data_col: str
            Column name for the data (single Weibull variable).

        Returns
        -------
        float
            The log-likelihood that this model generated the given data distribution.
        """
        def likelihood(x: Row) -> float:
            """likelihood for a single point"""
            censor = x[censor_col]
            data = x[data_col]
            return (self.pdf(data)**censor) * ((1. - self.cdf(data))**(1. - censor))
        return df.rdd.map(lambda x: math.log(likelihood(x))).sum()  # action
