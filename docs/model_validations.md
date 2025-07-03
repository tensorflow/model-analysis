# Tensorflow Model Analysis Model Validations

## Overview

TFMA supports validating a model by setting up value thresholds and change
thresholds based on the
[supported metrics](https://github.com/tensorflow/model-analysis/blob/master/g3doc/metrics.md).

## Configuration

### GenericValueThreshold

Value threshold is useful to gate the candidate model by checking whether the
corresponding metrics is larger than a lower bound and/or smaller than a upper
bound. User can set either one or both of the lower_bound and upper_bound
values. The lower_bound is default to negative infinity if unset, and the
upper_bound defaults to infinity if unset.

```python
import tensorflow_model_analysis as tfma

lower_bound = tfma.GenericValueThreshold(lower_bound={'value':0})
upper_bound = tfma.GenericValueThreshold(upper_bound={'value':1})
lower_upper_bound = tfma.GenericValueThreshold(lower_bound={'value':0},
                                               upper_bound={'value':1))
```

### GenericChangeThreshold

Change threhold is useful to gate the candidate model by checking whether the
corresponding metric is larger/smaller than that of a baseline model. There are
two ways that the change can be measured: absolute change and relative change.
Aboslute change is calculated as the value diference between the metrics of the
candidate and baseline model, namely, *v_c - v_b* where *v_c* denotes the
candidate metric value and *v_b* denotes the baseline value. Relative value is
the relative difference between the metric of the candidate and the baseline,
namely, *v_c/v_b*. The absolute and the relative threshold can co-exist to gate
model by both criteria. Besides setting up threshold values, user also need to
configure the MetricDirection. for metrics with favorably higher values (e.g.,
AUC), set the direction to HIGHER_IS_BETTER, for metrics with favorably lower
values (e.g., loss), set the direction to LOWER_IS_BETTER. Change thresholds
require a baseline model to be evaluated along with the candidate model. See
[Getting Started guide](https://github.com/tensorflow/model-analysis/blob/master/g3doc/get_started.md#model-validation)
for an example.

```python
import tensorflow_model_analysis as tfma

absolute_higher_is_better = tfma.GenericChangeThreshold(absolute={'value':1},
                                                        direction=tfma.MetricDirection.HIGHER_IS_BETTER)
absolute_lower_is_better = tfma.GenericChangeThreshold(absolute={'value':1},
                                                       direction=tfma.MetricDirection.LOWER_IS_BETTER)
relative_higher_is_better = tfma.GenericChangeThreshold(relative={'value':1},
                                                        direction=tfma.MetricDirection.HIGHER_IS_BETTER)
relative_lower_is_better = tfma.GenericChangeThreshold(relative={'value':1},
                                                       direction=tfma.MetricDirection.LOWER_IS_BETTER)
absolute_and_relative = tfma.GenericChangeThreshold(relative={'value':1},
                                                    absolute={'value':0.2},
                                                    direction=tfma.MetricDirection.LOWER_IS_BETTER)
```

### Putting things together

The following example combines value and change thresholds:

```python
import tensorflow_model_analysis as tfma

lower_bound = tfma.GenericValueThreshold(lower_bound={'value':0.7})
relative_higher_is_better =
    tfma.GenericChangeThreshold(relative={'value':1.01},
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER)
auc_threshold = tfma.MetricThreshold(value_threshold=lower_bound,
                                     change_threshold=relative_higher_is_better)
```

It might be more readable to write down the config in proto format:

```python
from google.protobuf import text_format

auc_threshold = text_format.Parse("""
  value_threshold { lower_bound { value: 0.6 } }
  change_threshold { relative { value: 1.01 } }
""", tfma.MetricThreshold())
```

The MetricThreshold can be set to gate on both model Training Time metrics
(either EvalSavedModel or Keras saved model) and Post Training metrics (defined
in TFMA config). For Training Time metrics, the thresholds are specified in the
tfma.MetricsSpec:

```python
metrics_spec = tfma.MetricSpec(thresholds={'auc': auc_threshold})
```

For post training metrics, thresholds are defined directly in the
tfma.MetricConfig:

```python
metric_config = tfma.MetricConfig(class_name='TotalWeightedExample',
                                  threshold=lower_bound)
```

Here is an example along with the other settings in the EvalConfig:

```python
# Run in a Jupyter Notebook.
from google.protobuf import text_format

eval_config = text_format.Parse("""
  model_specs {
    # This assumes a serving model with a "serving_default" signature.
    label_key: "label"
    example_weight_key: "weight"
  }
  metrics_spec {
    # Training Time metric thresholds
    thresholds {
      key: "auc"
      value: {
        value_threshold {
          lower_bound { value: 0.7 }
        }
        change_threshold {
          direction: HIGHER_IS_BETTER
          absolute { value: -1e-10 }
        }
      }
    }
    # Post Training metrics and their thesholds.
    metrics {
      # This assumes a binary classification model.
      class_name: "AUC"
      threshold {
        value_threshold {
          lower_bound { value: 0 }
        }
      }
    }
  }
  slicing_specs {}
  slicing_specs {
    feature_keys: ["age"]
  }
""", tfma.EvalConfig())

eval_shared_models = [
  tfma.default_eval_shared_model(
      model_name=tfma.CANDIDATE_KEY,
      eval_saved_model_path='/path/to/saved/candiate/model',
      eval_config=eval_config),
  tfma.default_eval_shared_model(
      model_name=tfma.BASELINE_KEY,
      eval_saved_model_path='/path/to/saved/baseline/model',
      eval_config=eval_config),
]

eval_result = tfma.run_model_analysis(
    eval_shared_models,
    eval_config=eval_config,
    # This assumes your data is a TFRecords file containing records in the
    # tf.train.Example format.
    data_location="/path/to/file/containing/tfrecords",
    output_path="/path/for/output")

tfma.view.render_slicing_metrics(eval_result)
tfma.load_validation_result(output_path)
```

## Output

In addition to the metrics file output by the evaluator, when validation is
used, an additional "validations" file is also output. The payload format is
[ValidationResult](https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis/proto/validation_result.proto).
The output will have "validation_ok" set to True when there are no failures.
When there are failures, information is provided about the associated metrics,
the thresholds, and the metric values that were observed. The following is an
example where the "weighted_examle_count" is failing a value threshold (1.5 is
not smaller than 1.0, thus the failure):

```proto
  validation_ok: False
  metric_validations_per_slice {
    failures {
      metric_key {
        name: "weighted_example_count"
        model_name: "candidate"
      }
      metric_threshold {
        value_threshold {
          upper_bound { value: 1.0 }
        }
      }
      metric_value {
        double_value { value: 1.5 }
      }
    }
  }
```
