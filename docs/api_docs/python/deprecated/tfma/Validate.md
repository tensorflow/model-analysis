<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.Validate" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.Validate

```python
tfma.Validate(
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

Performs validation of alternative evaluations.

#### Args:

*   <b>`extracts`</b>: PCollection of extracts.
*   <b>`alternatives`</b>: Dict of PTransforms (Extracts -> Evaluation) whose
    output will be compared for validation purposes (e.g. 'baseline' vs
    'candidate').
*   <b>`validators`</b>: List of validators for validating the output from
    running the alternatives. The Validation outputs produced by the validators
    will be merged into a single output. If there are overlapping output keys,
    later outputs will replace earlier outputs sharing the same key.

#### Returns:

Validation dict.
