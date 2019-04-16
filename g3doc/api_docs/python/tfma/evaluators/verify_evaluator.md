<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.evaluators.verify_evaluator" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.evaluators.verify_evaluator

```python
tfma.evaluators.verify_evaluator(
    evaluator,
    extractors
)
```

Defined in
[`evaluators/evaluator.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/evaluators/evaluator.py).

<!-- Placeholder for "Used in" -->

Verifies evaluator is matched with an extractor.

#### Args:

*   <b>`evaluator`</b>: Evaluator to verify.
*   <b>`extractors`</b>: Extractors to use in verification.

#### Raises:

*   <b>`ValueError`</b>: If an Extractor cannot be found for the Evaluator.
