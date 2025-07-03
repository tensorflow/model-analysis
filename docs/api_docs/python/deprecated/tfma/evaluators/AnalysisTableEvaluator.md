<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.evaluators.AnalysisTableEvaluator" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.evaluators.AnalysisTableEvaluator

```python
tfma.evaluators.AnalysisTableEvaluator(
    key=constants.ANALYSIS_KEY,
    run_after=extractor.LAST_EXTRACTOR_STAGE_NAME,
    include=None,
    exclude=None
)
```

Defined in
[`evaluators/analysis_table_evaluator.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/evaluators/analysis_table_evaluator.py).

<!-- Placeholder for "Used in" -->

Creates an Evaluator for returning Extracts data for analysis.

If both include and exclude are None then tfma.INPUT_KEY extracts will be
excluded by default.

#### Args:

*   <b>`key`</b>: Name to use for key in Evaluation output.
*   <b>`run_after`</b>: Extractor to run after (None means before any
    extractors).
*   <b>`include`</b>: Keys of extracts to include in output. Keys starting with
    '_' are automatically filtered out at write time.
*   <b>`exclude`</b>: Keys of extracts to exclude from output.

#### Returns:

Evaluator for collecting analysis data. The output is stored under the key
'analysis'.

#### Raises:

*   <b>`ValueError`</b>: If both include and exclude are used.
