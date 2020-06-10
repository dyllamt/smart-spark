from typing import List, Tuple

import math
import random


def inverted_weibull_cdf(y: float, shape: float, scale: float) -> float:
    # inverted failure cdf for a Weibull distribution
    return scale * ((- math.log(1 - y))**(1. / shape))


def generate_weibull_data(points: int, shape: float, scale: float, seed=None) -> List[float]:
    # generates a sample that follows a Weibull distribution
    random.seed(a=seed)
    data = [random.random() for i in range(points)]
    data = [inverted_weibull_cdf(i, shape, scale) for i in data]
    return data


def censor_weibull_data(data: List[float], event_modifier: float, seed=None) -> List[Tuple[float, float]]:
    # simulates observing a data point at a random time (we may observe before a failure occurs)
    # modify the observation times with event_modifier
    random.seed(a=seed)
    max_var = max(data)
    observations = [random.random() * max_var * event_modifier for i in data]
    censor = [1. if o > d else 0. for o, d in zip(observations, data)]
    data = [min(d, o) for d, o in zip(data, observations)]
    return list(zip(data, censor))