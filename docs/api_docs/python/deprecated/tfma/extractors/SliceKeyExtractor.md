<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.extractors.SliceKeyExtractor" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.extractors.SliceKeyExtractor

```python
tfma.extractors.SliceKeyExtractor(
    slice_spec=None,
    materialize=True
)
```

Defined in
[`extractors/slice_key_extractor.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/extractors/slice_key_extractor.py).

<!-- Placeholder for "Used in" -->

Creates an extractor for extracting slice keys.

The incoming Extracts must contain a FeaturesPredictionsLabels extract keyed by
tfma.FEATURES_PREDICTIONS_LABELS_KEY. Typically this will be obtained by calling
the PredictExtractor.

The extractor's PTransform yields a copy of the Extracts input with an
additional extract pointing at the list of SliceKeyType values keyed by
tfma.SLICE_KEY_TYPES_KEY. If materialize is True then a materialized version of
the slice keys will be added under the key tfma.SLICE_KEYS_KEY.

#### Args:

*   <b>`slice_spec`</b>: Optional list of SingleSliceSpec specifying the slices
    to slice the data into. If None, defaults to the overall slice.
*   <b>`materialize`</b>: True to add MaterializedColumn entries for the slice
    keys.

#### Returns:

Extractor for slice keys.
