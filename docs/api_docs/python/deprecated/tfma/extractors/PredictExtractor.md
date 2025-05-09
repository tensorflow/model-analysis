<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.extractors.PredictExtractor" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.extractors.PredictExtractor

```python
tfma.extractors.PredictExtractor(
    eval_shared_model,
    desired_batch_size=None,
    materialize=True
)
```

Defined in
[`extractors/predict_extractor.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/extractors/predict_extractor.py).

<!-- Placeholder for "Used in" -->

Creates an Extractor for TFMAPredict.

The extractor's PTransform loads and runs the eval_saved_model against every
example yielding a copy of the Extracts input with an additional extract of type
FeaturesPredictionsLabels keyed by tfma.FEATURES_PREDICTIONS_LABELS_KEY.

#### Args:

*   <b>`eval_shared_model`</b>: Shared model parameters for EvalSavedModel.
*   <b>`desired_batch_size`</b>: Optional batch size for batching in Aggregate.
*   <b>`materialize`</b>: True to call the FeatureExtractor to add
    MaterializedColumn entries for the features, predictions, and labels.

#### Returns:

Extractor for extracting features, predictions, labels, and other tensors during
predict.
