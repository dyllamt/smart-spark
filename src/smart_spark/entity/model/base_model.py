"""Implementation of parametric models with computed properties.
"""
from typing import Dict
from abc import ABC, abstractmethod

from pyspark.rdd import RDD


class BaseModel(ABC):
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
