# Configuring an Eval Saved Model

TensorFlow Model Analysis (TFMA) can export a model's *evaluation graph* to a
special `SavedModel` called `EvalSavedModel`. (Note that the evaluation graph is
used and not the graph for training or inference.) The `EvalSavedModel` contains
additional information that allows TFMA to compute the same evaluation metrics
defined in the model in a distributed manner over a large amount of data and
user-defined slices.

## Modify an existing model

To use an existing model with TFMA, first modify the model to export the
`EvalSavedModel`. This is done by adding a call to
`tfma.export.export_eval_savedmodel` and is similar to
`estimator.export_savedmodel`. For example:

```python
# Define, train and export your estimator as usual
estimator = tf.estimator.DNNClassifier(...)
estimator.train(...)
estimator.export_savedmodel(...)

# Also export the EvalSavedModel
tfma.export.export_eval_savedmodel(
  estimator=estimator, export_dir_base=export_dir,
  eval_input_receiver_fn=eval_input_receiver_fn)
```

`eval_input_receiver_fn` must be defined and is similar to the
`serving_input_receiver_fn` for `estimator.export_savedmodel`. Like
`serving_input_receiver_fn`, the `eval_input_receiver_fn` function defines an
input placeholder example, parses the features from the example, and returns the
parsed features. It parses and returns the label.

The following snippet defines an example `eval_input_receiver_fn`:

```python
country = tf.feature_column.categorical_column_with_hash('country', 100)
language = tf.feature_column.categorical_column_with_hash('language', 100)
age = tf.feature_column.numeric_column('age')
label = tf.feature_column.numeric_column('label')

def eval_input_receiver_fn():
  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_placeholder')

  # This *must* be a dictionary containing a single key 'examples', which
  # points to the input placeholder.
  receiver_tensors = {'examples': serialized_tf_example}

  feature_spec =  tf.feature_column.make_parse_example_spec(
      [country, language, age, label])
  features = tf.io.parse_example(serialized_tf_example, feature_spec)

  return tfma.export.EvalInputReceiver(
    features=features,
    receiver_tensors=receiver_tensors,
    labels=features['label'])
```

In this example you can see that:

*   `labels` can also be a dictionary. Useful for a multi-headed model.
*   The `eval_input_receiver_fn` function will, most likely, be the same as your
    `serving_input_receiver_fn` function. But, in some cases, you may want to
    define additional features for slicing. For example, you introduce an
    `age_category` feature which divides the `age` feature into multiple
    buckets. You can then slice on this feature in TFMA to help understand how
    your model's performance differs across different age categories.

## Adding Post Export Metrics

Additional metrics that are not included in the model can be aded using
`add_metrics_callbacks`. For more details, see the Python help for
`run_model_analysis`.

## End-to-end examples

Try the extensive
[end-to-end example](https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline)
featuring [TensorFlow Transform](https://github.com/tensorflow/transform) for
feature preprocessing,
[TensorFlow Estimators](https://www.tensorflow.org/guide/estimators) for
training,
[TensorFlow Model Analysis](https://github.com/tensorflow/model-analysis) and
Jupyter for evaluation, and
[TensorFlow Serving](https://github.com/tensorflow/serving) for serving.

## Adding a Custom Post Export Metric

If you want to add your own custom post export metric in TFMA, please checkout
the documentation
[here](https://github.com/tensorflow/model-analysis/blob/master/g3doc/post_export_metrics.md).
