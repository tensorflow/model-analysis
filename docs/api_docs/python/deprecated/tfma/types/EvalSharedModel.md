<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.types.EvalSharedModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="model_path"/>
<meta itemprop="property" content="add_metrics_callbacks"/>
<meta itemprop="property" content="include_default_metrics"/>
<meta itemprop="property" content="example_weight_key"/>
<meta itemprop="property" content="additional_fetches"/>
<meta itemprop="property" content="shared_handle"/>
<meta itemprop="property" content="construct_fn"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfma.types.EvalSharedModel

## Class `EvalSharedModel`

### Aliases:

*   Class `tfma.EvalSharedModel`
*   Class `tfma.types.EvalSharedModel`

Defined in
[`types.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/types.py).

<!-- Placeholder for "Used in" -->

Shared model used during extraction and evaluation.

#### Attributes:

*   <b>`model_path`</b>: Path to EvalSavedModel (containing the saved_model.pb
    file).
*   <b>`add_metrics_callbacks`</b>: Optional list of callbacks for adding
    additional metrics to the graph. The names of the metrics added by the
    callbacks should not conflict with existing metrics. See below for more
    details about what each callback should do. The callbacks are only used
    during evaluation.
*   <b>`include_default_metrics`</b>: True to include the default metrics that
    are part of the saved model graph during evaluation.
*   <b>`example_weight_key`</b>: Deprecated.
*   <b>`additional_fetches`</b>: Prefixes of additional tensors stored in
    signature_def.inputs that should be fetched at prediction time. The
    "features" and "labels" tensors are handled automatically and should not be
    included in this list.
*   <b>`shared_handle`</b>: Optional handle to a shared.Shared object for
    sharing the in-memory model within / between stages.
*   <b>`construct_fn`</b>: A callable which creates a construct function to set
    up the tensorflow graph. Callable takes a beam.metrics distribution to track
    graph construction time.

More details on add_metrics_callbacks:

Each add_metrics_callback should have the following prototype: def
add_metrics_callback(features_dict, predictions_dict, labels_dict):

Note that features_dict, predictions_dict and labels_dict are not necessarily
dictionaries - they might also be Tensors, depending on what the model's
eval_input_receiver_fn returns.

It should create and return a metric_ops dictionary, such that
metric_ops['metric_name'] = (value_op, update_op), just as in the Trainer.

Short example:

def add_metrics_callback(features_dict, predictions_dict, labels): metrics_ops =
{} metric_ops['mean_label'] = tf.metrics.mean(labels)
metric_ops['mean_probability'] = tf.metrics.mean(tf.slice(
predictions_dict['probabilities'], [0, 1], [2, 1])) return metric_ops

<h2 id="__new__"><code>__new__</code></h2>

```python
@staticmethod
__new__(
    cls,
    model_path=None,
    add_metrics_callbacks=None,
    include_default_metrics=True,
    example_weight_key=None,
    additional_fetches=None,
    shared_handle=None,
    construct_fn=None
)
```

Create new instance of EvalSharedModel(model_path, add_metrics_callbacks,
include_default_metrics, example_weight_key, additional_fetches, shared_handle,
construct_fn)

## Properties

<h3 id="model_path"><code>model_path</code></h3>

<h3 id="add_metrics_callbacks"><code>add_metrics_callbacks</code></h3>

<h3 id="include_default_metrics"><code>include_default_metrics</code></h3>

<h3 id="example_weight_key"><code>example_weight_key</code></h3>

<h3 id="additional_fetches"><code>additional_fetches</code></h3>

<h3 id="shared_handle"><code>shared_handle</code></h3>

<h3 id="construct_fn"><code>construct_fn</code></h3>
