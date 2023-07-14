# Smart Spark

Processing for S.M.A.R.T. hard-drive statistics using the SPARK framework.

## Installation

```
pip install -e .
```


## Contents

### Backblaze Dataset

The backblaze dataset for hard drive statistics contains diagnostic data from their data center. The
dataset is available at the following url: <https://www.backblaze.com/b2/hard-drive-test-data.html>

```
from smart_spark.schema import schema as bb_schema
```

### Weibull Distribution

The Weibull distribution is a parametric distribution that is known to describe failure
distributions well. The distribution parameters can be estimated from censored or un-cencored data by
minimizing the log likelihood of the distribution, given the data.

```
from smart_spark.weibull import weibull
```

### Pyspark Jobs

The following data pipeline loads the backblaze dataset, filters the dataset for failure events,
estimates the distribution parameters for each hard drive model, and returns the distribution paramters
as well as bin counts of the underlying data.

```
from smart_spark.jobs.preprocessing import failure_events, partition_by_model
from smart_spark.jobs.optimization import fit_censored_weibull
from smart_spark.jobs.postprocessing import bin_counts, numeric_cdf
```