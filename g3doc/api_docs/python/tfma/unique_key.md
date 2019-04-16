<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.unique_key" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.unique_key

```python
tfma.unique_key(
    key,
    current_keys,
    update_keys=False
)
```

Defined in
[`util.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/util.py).

<!-- Placeholder for "Used in" -->

Returns a unique key given a list of current keys.

If the key exists in current_keys then a new key with _1, _2, ..., etc appended
will be returned, otherwise the key will be returned as passed.

#### Args:

*   <b>`key`</b>: desired key name.
*   <b>`current_keys`</b>: List of current key names.
*   <b>`update_keys`</b>: True to append the new key to current_keys.
