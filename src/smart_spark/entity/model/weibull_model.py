"""Implementation of weibull states.
"""
import math
from pyspark.rdd import RDD

from .base_model import BaseModel


class WeibullModel(BaseModel):
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
        """The probability density function over the single variable.

        Parameters
        ----------
        x: float
            The single variable.

        Returns
        -------
        float
            The probability density.
        """
        shape = self.parameters["shape"]
        scale = self.parameters["scale"]
        return (shape / scale) * ((x / scale)**(shape - 1)) * math.exp(-((x / scale)**shape))

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
