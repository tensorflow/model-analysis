## Post Export Metrics

As the name suggests, this is a metric that is added post-export, before
evaluation.

TFMA is packaged with several pre-defined evaluation metrics, like
example_count, auc, confusion_matrix_at_thresholds, precision_recall_at_k, mse,
mae, to name a few. (Complete list
[here](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/metrics/__init__.py).)

If you don’t find an existing metrics relevant to your use-case, or want to
customize a metric, you can define your own custom metric. Read on for the
details!

## Adding Custom Metrics in TFMA

### Defining Custom Metrics in TFMA 1.x

Tip: For code references, please check metrics like
[FairnessIndicators](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/addons/fairness/post_export_metrics/fairness_indicators.py),
[MeanAbsoluteError](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py#L1866)
etc.

##### Extend Abstract Base Class

To add a custom metric, create a new class extending
[\_PostExportMetric](https://github.com/tensorflow/model-analysis/blob/ed6cd1e110d1218bd433f77b37d734efe6a227e9/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py#L338)
abstract class and define its constructor and implement abstract / unimplemented
methods.

##### Define Constructor

In the constructor, take as parameters all the relevant information like
label_key, prediction_key, example_weight_key, metric_tag, etc. required for
custom metric.

##### Implement Abstract / Unimplemented Methods

*   [check_compatibility](https://github.com/tensorflow/model-analysis/blob/ed6cd1e110d1218bd433f77b37d734efe6a227e9/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py#L477)

    Implement this method to check for compatibility of the metric with the
    model being evaluated, i.e. checking if all required features, expected
    label and prediction key are present in the model in appropriate data type.
    It takes three arguments:

    *   features_dict
    *   predictions_dict
    *   labels_dict

    These dictionaries contains references to Tensors for the model.

*   [get_metric_ops](https://github.com/tensorflow/model-analysis/blob/ed6cd1e110d1218bd433f77b37d734efe6a227e9/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py#L500)

    Implement this method to provide metric ops (value and update ops) to
    compute the metric. Similar to check_compatibility method, it also takes
    three arguments:

    *   features_dict
    *   predictions_dict
    *   labels_dict

    Define your metric computation logic using these references to Tensors for
    the model.

    Note: Result of update_ops must be additive for TFMA to sum up per worker
    update_op output. Check out examples
    [here](https://www.tensorflow.org/api_docs/python/tf/compat/v1/metrics/).

*   [populate_stats_and_pop](https://github.com/tensorflow/model-analysis/blob/ed6cd1e110d1218bd433f77b37d734efe6a227e9/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py#L524)
    and
    [populate_plots_and_pop](https://github.com/tensorflow/model-analysis/blob/ed6cd1e110d1218bd433f77b37d734efe6a227e9/tensorflow_model_analysis/post_export_metrics/post_export_metrics.py#L542)

    Implement this metric to convert raw metric results to
    [MetricValue](https://github.com/tensorflow/model-analysis/blob/ed6cd1e110d1218bd433f77b37d734efe6a227e9/tensorflow_model_analysis/proto/metrics_for_slice.proto#L197)
    and
    [PlotData](https://github.com/tensorflow/model-analysis/blob/ed6cd1e110d1218bd433f77b37d734efe6a227e9/tensorflow_model_analysis/proto/metrics_for_slice.proto#L323)
    proto format. This takes three arguments:

    *   slice_key: Name of slice metric belongs to.
    *   combined_metrics: Dictionary containing raw results.
    *   output_metrics: Output dictionary containing metric in desired proto
        format.

```
@_export('my_metric')
class _MyMetric(_PostExportMetric):
   def __init__(self,
                target_prediction_keys: Optional[List[Text]] = None,
                labels_key: Optional[Text] = None,
                metric_tag: Optional[Text] = None):
      self._target_prediction_keys = target_prediction_keys
      self._label_keys = label_keys
      self._metric_tag = metric_tag
      self._metric_key = 'my_metric_key'

   def check_compatibility(self, features_dict:types.TensorTypeMaybeDict,
                           predictions_dict: types.TensorTypeMaybeDict,
                           labels_dict: types.TensorTypeMaybeDict) -> None:
       # Add compatibility check needed for the metric here.

   def get_metric_ops(self, features_dict: types.TensorTypeMaybeDict,
                      predictions_dict: types.TensorTypeMaybeDict,
                      labels_dict: types.TensorTypeMaybeDict
                     ) -> Dict[bytes, Tuple[types.TensorType,
                     types.TensorType]]:
        # Metric computation logic here.
        # Define value and update ops.
        value_op = compute_metric_value(...)
        update_op = create_update_op(... )
        return {self._metric_key: (value_op, update_op)}

   def populate_stats_and_pop(
       self, slice_key: slicer.SliceKeyType, combined_metrics: Dict[Text, Any],
       output_metrics: Dict[Text, metrics_pb2.MetricValue]) -> None:
       # Parses the metric and converts it into required metric format.
       metric_result = combined_metrics[self._metric_key]
       output_metrics[self._metric_key].double_value.value = metric_result
```

### Usage

```
# Custom metric callback
custom_metric_callback = my_metric(
    labels_key='label',
    target_prediction_keys=['prediction'])

fairness_indicators_callback =
   post_export_metrics.fairness_indicators(
        thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], labels_key=label)

add_metrics_callbacks = [custom_metric_callback,
   fairness_indicators_callback]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=eval_saved_model_path,
    add_metrics_callbacks=add_metrics_callbacks)

eval_config = tfma.EvalConfig(...)

# Run evaluation
tfma.run_model_analysis(
    eval_config=eval_config, eval_shared_model=eval_shared_model)
```
