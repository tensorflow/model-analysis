<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.extractors.Filter" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.extractors.Filter

```python
tfma.extractors.Filter(
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

Filters extracts to include/exclude specified keys.

#### Args:

*   <b>`extracts`</b>: PCollection of extracts.
*   <b>`include`</b>: Keys to include in output.
*   <b>`exclude`</b>: Keys to exclude from output.

#### Returns:

Filtered PCollection of Extracts.

#### Raises:

*   <b>`ValueError`</b>: If both include and exclude are used.
