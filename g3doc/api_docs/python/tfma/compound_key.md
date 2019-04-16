<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.compound_key" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.compound_key

```python
tfma.compound_key(
    keys,
    separator=KEY_SEPARATOR
)
```

Defined in
[`util.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/util.py).

<!-- Placeholder for "Used in" -->

Returns a compound key based on a list of keys.

#### Args:

*   <b>`keys`</b>: Keys used to make up compound key.
*   <b>`separator`</b>: Separator between keys. To ensure the keys can be parsed
    out of any compound key created, any use of a separator within a key will be
    replaced by two separators.
