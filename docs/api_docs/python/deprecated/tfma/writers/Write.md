<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.writers.Write" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.writers.Write

```python
tfma.writers.Write(
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

Writes given Evaluation or Validation data using given writer PTransform.

#### Args:

*   <b>`evaluation_or_validation`</b>: Evaluation or Validation data.
*   <b>`key`</b>: Key for Evaluation or Validation output to write. It is valid
    for the key to not exist in the dict (in which case the write is a no-op).
*   <b>`ptransform`</b>: PTransform to use for writing.

#### Raises:

*   <b>`ValueError`</b>: If Evaluation or Validation is empty. The key does not
    need to exist in the Evaluation or Validation, but the dict must not be
    empty.

#### Returns:

beam.pvalue.PDone.
