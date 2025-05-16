<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.WriteResults" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.WriteResults

```python
tfma.WriteResults(
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

Writes Evaluation or Validation results using given writers.

#### Args:

*   <b>`evaluation_or_validation`</b>: Evaluation or Validation output.
*   <b>`writers`</b>: Writes to use for writing out output.

#### Raises:

*   <b>`ValueError`</b>: If Evaluation or Validation is empty.

#### Returns:

beam.pvalue.PDone.
