<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.export.build_parsing_eval_input_receiver_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.export.build_parsing_eval_input_receiver_fn

```python
tfma.export.build_parsing_eval_input_receiver_fn(
    feature_spec,
    label_key
)
```

Defined in
[`eval_saved_model/export.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/eval_saved_model/export.py).

<!-- Placeholder for "Used in" -->

Build a eval_input_receiver_fn expecting fed tf.Examples.

Creates a eval_input_receiver_fn that expects a serialized tf.Example fed into a
string placeholder. The function parses the tf.Example according to the provided
feature_spec, and returns all parsed Tensors as features.

#### Args:

*   <b>`feature_spec`</b>: A dict of string to
    `VarLenFeature`/`FixedLenFeature`.
*   <b>`label_key`</b>: The key for the label column in the feature_spec. Note
    that the label must be part of the feature_spec. If None, does not pass a
    label to the EvalInputReceiver (note that label_key must be None and not
    simply the empty string for this case).

#### Returns:

A eval_input_receiver_fn suitable for use with TensorFlow model analysis.
