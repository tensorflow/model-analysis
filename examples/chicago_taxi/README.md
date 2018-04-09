# Chicago Taxi Example

The Chicago Taxi example demonstrates the end-to-end workflow and steps of how
to transform data, train a model, analyze and serve it, using:

* [TensorFlow Transform](https://github.com/tensorflow/transform) for
feature preprocessing
* TensorFlow [Estimators](https://www.tensorflow.org/programmers_guide/estimators)
for training
* [TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis) and
Jupyter for evaluation
* [TensorFlow Serving](https://github.com/tensorflow/serving) for serving

The example shows two modes of deployment.

* The first is a “local mode” with all necessary dependencies and components
deployed locally.
* The second is a “cloud mode”, where all components will be deployed on Google
Cloud.

In the future we will be showing additional deployment modes, so dear reader,
feel free to check back in periodically!

# Table Of Contents
1. [Dataset](#dataset)
1. [Local Prerequisites](#local-prerequisites)
1. [Running the Local Example](#running-the-local-example)
1. [Cloud Prerequisites](#cloud-prerequisites)
1. [Running the Cloud Example](#running-the-cloud-example)

# Dataset

In this example you will use the Taxi Trips [dataset](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)
released by the City of Chicago.

*Disclaimer: This site provides applications using data that has been modified
for use from its original source, www.cityofchicago.org, the official website of
the City of Chicago.  The City of Chicago makes no claims as to the content,
accuracy, timeliness, or completeness of any of the data provided at this site.
The data provided at this site is subject to change at any time.  It is
understood that the data provided at this site is being used at one’s own risk.*

You can [read more](https://cloud.google.com/bigquery/public-data/chicago-taxi)
about the dataset in [Google BigQuery](https://cloud.google.com/bigquery/), and
explore the full dataset in its
[UI](https://bigquery.cloud.google.com/dataset/bigquery-public-data:chicago_taxi_trips).

# Local Prerequisites

This example relies on [Apache Beam](https://beam.apache.org/) for its
distributed processing. The Apache Beam SDK ([BEAM-1373](https://issues.apache.org/jira/browse/BEAM-1373))
and TensorFlow Serving API ([Issue-700](https://github.com/tensorflow/serving/issues/700))
are not yet available for Python 3, and as such the example requires Python 2.7
.

## Install Dependencies

We will be doing all development within an isolated Python Virtual Environment.
This allows us to isolate your environment as you experiment with different
versions of dependencies.

There are many ways to install virtualenv on different platforms, we show two
versions here:

* On Linux:
```
sudo apt-get install python-pip python-dev build-essential
```
* On Mac:
```
sudo easy_install pip
```

Next we enforce Python 2.7 (which this example requires), ensure we have the
latest versions of pip and virtualenv, create a virtualenv called “taxi” and
switch into it:

```
alias python=python2.7
sudo pip install --upgrade pip
sudo pip install --upgrade virtualenv
python -m virtualenv taxi
source ./taxi/bin/activate
```

Next we install dependencies required by the Chicago Taxi example.

```
pip install -r requirements.txt
```

Next we register our TensorFlow Model Analysis rendering components with
Jupyter Notebook:

```
jupyter nbextension install --py --symlink --sys-prefix tensorflow_model_analysis
jupyter nbextension enable --py --sys-prefix tensorflow_model_analysis
```

# Running the Local Example

The benefit of the local example is that you can edit any part of the pipeline
and experiment very quickly with various components. The example comes with a
small subset of the Taxi Trips dataset as CSV files.

## Preprocessing with TensorFlow Transform

tf.Transform (`preprocess.py`) allows us to do preprocessing using results of
full-pass operations over the dataset. To run this step locally, simply call:

```
bash ./preprocess_local.sh
```

We first ask tf.Transform to compute global statistics (mean, standard dev,
bucket cutoffs, etc) in our `preprocessing_fn()`:

* We take dense float features such as trip distance and time, compute the
global mean and standard deviations, and scale features to their z scores.
This allows the SGD optimizer to treat steps in different directions more
uniformly.
* We create a finite vocabulary for string features, which will allow us to
treat them as categoricals in our model.
* We bucket our longitude/latitude features, which allows the model to fit
multiple weights to different parts of the lat/long grid.
* We include time-of-day and day-of-week features, which will allow the model to
more easily pick up on seasonality.
* Our label is binary; 1 if the tip is more than 20% of the fare, 0 if it is
lower.

Preprocess creates TensorFlow Operations for applying the transforms and leaves
TensorFlow **Placeholders** in the graph for the mean, bucket cutoffs, etc.

We call tf.Transform’s `Analyze()` function, which builds the MapReduce-style
callbacks to compute these statistics across the entire data set.
**Apache Beam** allows applications to write such data-transforming code once,
and handles the job of placing the code onto workers, whether they are in the
cloud, across an enterprise cluster or on the local machine.  In this part of
our example, we use `DirectRunner` (see `preprocess_local.sh`), to request that
our code all runs on the local machine.  At the end of the Analyze job, the
**Placeholders** are replaced with their respective statistics (mean, standard
deviation, etc).

Notice that tf.Transform is also used to shuffle the examples
(`beam.transforms.Reshuffle`) -- this is very important for the efficiency of
non-convex stochastic learners such as SGD with Deep Neural Networks.

Finally, we call `WriteTransformFn()` to save our transform and `TransformDataset()`
to materialize our examples for training. There are two key outputs of the
Preprocessing step:

* SavedModel containing the transform graph
* Materialized, transformed examples in compressed TFRecord files (these will be
inputs to the TensorFlow trainer)

## Model Training

In the next step we train our model using TensorFlow. To run this step locally,
simply call:

```
bash ./train_local.sh
```

Our model leverages TensorFlow’s
[Estimators](https://www.tensorflow.org/programmers_guide/estimators), which is
built inside of `build_estimator()` in `model.py`. The trainer takes as input
materialized, transformed examples from the previous step. Notice our pattern of
sharing schema information between preprocessing and training using `taxi.py`
to avoid redundant code.

The `input_fn()` builds a parser that takes in tf.Example protos from the
materialized TFRecord file emitted in our previous Preprocessing step.
It also applies the tf.Transform preprocessing operations built in the
preprocessing step. Also, we must not forget to remove our label, as we wouldn’t
want our model to treat it as a feature!

During training, `feature_columns` also come into play, which tell the model how
to use features: for example, vocabulary features are fed into the model with
`categorical_column_with_identity()`, which tells our model to logically treat
this feature as a one-hot encoded feature.

Inside the `eval_input_receiver_fn()` callback we emit a TensorFlow graph which
parses raw examples, identifies the features and label and applies the
tf.Transform graph that will be used in the TensorFlow Model Analysis batch job.

Finally, our model emits a graph suitable for serving -- we will show how to
serve this model below.

Notice also that the trainer runs a quick evaluation at the end of the batch
job. This limited evaluation can run only on a single machine -- we will use
TensorFlow Model Analysis, which leverages Apache Beam, for distributed
evaluation below.

To recap, our trainer outputs the following:

* SavedModel containing the serving graph (for use with TensorFlow Serving)
* SavedModel containing the evaluation graph (for use with TensorFlow Model
Analysis)

## TensorFlow Model Analysis Evaluation Batch Job

Next we run a batch job to evaluate our model against the entire data set. To
run the batch evaluator:

```
bash ./process_tfma_local.sh
```

As a reminder, we are running this step locally with a small CSV dataset.
TensorFlow Model Analysis will do a full pass over this dataset and compute
metrics. We will demonstrate running this step as a distributed job over a much
larger data set in the cloud section below.

Like tf.Transform, tf.ModelAnalysis leverages Apache Beam to run the distributed
computation.  Also, our evaluation takes in raw examples *not* transformed
examples -- this means that our (local) input here is a CSV file and that we are
using the SavedModel from the previous step to parse, apply the tf.Transform
graph and run the model.  For completeness, it is worth noting that it is also
possible to analyze our models in terms of transformed features rather than raw
features -- we leave this as a user exercise for this example.

Also notice in `process_tfma.py`, we specified a `slice_spec`. This tells the
tf.ModelAnalysis job which slices we are interested in visualizing (in the next
step). Slices are subsets of our data based on feature values. tf.ModelAnalysis
computes metrics for each of those slices.

This job outputs a file that can be visualized in Jupyter in the
`tf.model_analysis` renderer in our next stage.

## Looking at TensorFlow Model Analysis Rendered Metrics

We will look at our sliced results using a Jupyter notebook. To run the Jupyter
notebook locally:

```
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
```

The above will give you an URL of the form `http://0.0.0.0:8888/?token=...`
The first time you go that page, you might need to enter a username and password
(you can enter anything e.g. ‘test’ for both).

From the files tab, open `chicago_taxi_tfma.ipynb`. Follow the instructions in
the notebook until you can visualize the slicing metrics browser with
`tfma.view.render_slicing_metrics` and the time series graph with
`tfma.view.render_time_series`.
TODO:screenshots in images/ directory

Note that this notebook is completely self-contained and does not rely on the
prior scripts having been already run.

## Serving Your TensorFlow Model

Next we will serve the model we just created using TensorFlow Serving.  In our
example, we run our server in a Docker container that we will run locally.
Information for installing Docker locally can be found at
https://docs.docker.com/install.

To start the server, open a separate terminal and run the serving script:

```
bash ./start_model_server_local.sh
```

The script will build a Docker image and then start the TensorFlow Model Server
within a running container listening on localhost port 9000 for gRPC requests.
The Model server will load the model exported from our Trainer step above.

To send a request to the server and run model inference, run:

```
bash ./classify_local.sh
```

More information about serving can be found in the
[TensorFlow Serving Documentation](https://www.tensorflow.org/serving/).

## Playground Notebook

We also offer `chicago_taxi_tfma_local_playground.ipynb`, a notebook that calls
the same scripts above. It contains a more detailed description of the APIs,
like custom metrics and plots, and the UI components.

Note that this notebook is completely self-contained and does not rely on the
prior scripts having been already run.

# Cloud Prerequisites

We rely heavily upon the local prerequisites [above](#local-prerequisites) and
add a few more requirements for [Gooogle Cloud Platform](https://cloud.google.com/).

Make sure you follow the [Google Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/)
setup [here](https://cloud.google.com/ml-engine/docs/how-tos/getting-set-up) and
the [Google Cloud Dataflow](https://cloud.google.com/dataflow/) setup
[here](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)
before trying the example.

The speed of execution of the example might be limited by default
[Google Compute Engine](https://cloud.google.com/compute) quota.
We recommend sufficient quota for approximately 250 Dataflow VMs which amounts
to: **250 CPUs, 250 IP Addresses and 62500 GB of Persistent Disk**. For more
details please see [GCE Quota](https://cloud.google.com/compute/quotas) and
[Dataflow Quota](https://cloud.google.com/dataflow/quotas) documentation.

Our example will use [Google Cloud Storage](https://cloud.google.com/storage/)
Buckets to store data and local environment variables to pass paths from job to
job -- as such the jobs should be run from a single command-line shell.

Authenticate and switch to your project:

```
gcloud auth application-default login
gcloud config set project $PROJECT_NAME
```

To create our gs:// bucket:

```
export MYBUCKET=gs://$(gcloud config list --format 'value(core.project)')-chicago-taxi
gsutil mb $MYBUCKET
```

Make sure you are inside the virtualenv we created above (you will need to do
these steps whenever re-enter your shell):

```
source ./bin/taxi/activate
```

# Running the Cloud Example

Next we will run in the Taxi Trips example in the cloud.  Unlike our local
example, our input will be a much larger dataset, hosted on Google BigQuery.

## Preprocessing with TensorFlow Transform on Google Cloud Dataflow

We will now use the same code from our local tf.Transform (in `preprocess.py`)
to do our distributed transform. To start the job:

```
source ./preprocess_dataflow.sh
```

You can watch the status of your running job at
https://console.cloud.google.com/dataflow

In this case we are reading data from Google BigQuery rather than from a small,
local CSV. Also unlike our local example above, we are using the
`DataflowRunner`, which starts distributed processing over several workers in
the cloud.

Our outputs will be the same as in the local job, but stored on Google Cloud
Storage:

* SavedModel containing the transform graph:
```
gsutil ls $TFT_OUTPUT_PATH/transform_fn
```
* Materialized, transformed examples (train_transformed-\*):
```
gsutil ls $TFT_OUTPUT_PATH
```

## Model Training on Google Cloud Machine Learning Engine

We next run the distributed TensorFlow trainer in the cloud:

```
source ./train_mlengine.sh
```

You can find your running job and its status here:
https://console.cloud.google.com/mlengine

Notice that while our trainer is running in the cloud, it is using ML Engine,
not Dataflow, to the distributed computation. Again, our outputs are identical
to the local run:

* SavedModel containing the serving graph (for use with TensorFlow Serving):
```
gsutil ls $WORKING_DIR/serving_model_dir/export/chicago-taxi
```
* SavedModel containing the evaluation graph (for use with TensorFlow Model
Analysis):
```
gsutil ls $WORKING_DIR/eval_model_dir
```

## Model Evaluation with TensorFlow Model Analysis on Google Cloud Dataflow

We next run a distributed batch job to compute sliced metrics across the large
Google BigQuery data set. In this step, tf.ModelAnalysis takes advantage of the
`DataflowRunner` to control its workers. You can run this job as follows:

```
source ./process_tfma_dataflow.sh
```

Our output will be the `eval_result` file that will be rendered by our notebook
in the next step:

```
gsutil ls -l $TFT_OUTPUT_PATH/eval_result_dir
```

## Rendering TensorFlow Model Analysis Results in Local Jupyter Notebook

The steps for looking at your results in a notebook are identical to the ones from
the local job. Simply go to `chicago_taxi_tfma.ipynb` notebook,
and set up the output directory to see the results.

## Model Serving on Google Cloud Machine Learning Engine

Finally, we serve the model we created in the training step in the cloud. To
start serving your model from the cloud, you can run:

```
bash ./start_model_server_mlengine.sh
```

To send a request to the cloud:

```
bash ./classify_mlengine.sh
```
