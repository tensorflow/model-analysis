<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="ANALYSIS_KEY"/>
<meta itemprop="property" content="ATTRIBUTIONS_KEY"/>
<meta itemprop="property" content="BASELINE_KEY"/>
<meta itemprop="property" content="BASELINE_SCORE_KEY"/>
<meta itemprop="property" content="CANDIDATE_KEY"/>
<meta itemprop="property" content="DATA_CENTRIC_MODE"/>
<meta itemprop="property" content="EXAMPLE_SCORE_KEY"/>
<meta itemprop="property" content="FEATURES_KEY"/>
<meta itemprop="property" content="FEATURES_PREDICTIONS_LABELS_KEY"/>
<meta itemprop="property" content="INPUT_KEY"/>
<meta itemprop="property" content="LABELS_KEY"/>
<meta itemprop="property" content="METRICS_KEY"/>
<meta itemprop="property" content="MODEL_CENTRIC_MODE"/>
<meta itemprop="property" content="PLOTS_KEY"/>
<meta itemprop="property" content="PREDICTIONS_KEY"/>
<meta itemprop="property" content="VERSION"/>
</div>

# Module: tfma

Defined in
[`__init__.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/__init__.py).

<!-- Placeholder for "Used in" -->

Init module for TensorFlow Model Analysis.

## Modules

[`constants`](./tfma/constants.md) module: Constants used in TensorFlow Model
Analysis.

[`evaluators`](./tfma/evaluators.md) module: Init module for TensorFlow Model
Analysis evaluators.

[`export`](./tfma/export.md) module: Library for exporting the EvalSavedModel.

[`exporter`](./tfma/exporter.md) module: `Exporter` class represents different
flavors of model export.

[`extractors`](./tfma/extractors.md) module: Init module for TensorFlow Model
Analysis extractors.

[`post_export_metrics`](./tfma/post_export_metrics.md) module: Library
containing helpers for adding post export metrics for evaluation.

[`types`](./tfma/types.md) module: Types.

[`validators`](./tfma/validators.md) module: Init module for TensorFlow Model
Analysis validators.

[`version`](./tfma/version.md) module: Contains the version string for this
release of TFMA.

[`view`](./tfma/view.md) module: Initializes TFMA's view rendering api.

[`writers`](./tfma/writers.md) module: Init module for TensorFlow Model Analysis
writers.

## Classes

[`class EvalConfig`](./tfma/EvalConfig.md): Config used for extraction and
evaluation.

[`class EvalResult`](./tfma/EvalResult.md): EvalResult(slicing_metrics, plots,
config)

[`class EvalSharedModel`](./tfma/types/EvalSharedModel.md): Shared model used
during extraction and evaluation.

[`class FeaturesPredictionsLabels`](./tfma/types/FeaturesPredictionsLabels.md):
FeaturesPredictionsLabels(input_ref, features, predictions, labels)

[`class MaterializedColumn`](./tfma/types/MaterializedColumn.md):
MaterializedColumn(name, value)

## Functions

[`ExtractAndEvaluate(...)`](./tfma/ExtractAndEvaluate.md): Performs Extractions
and Evaluations in provided order.

[`ExtractEvaluateAndWriteResults(...)`](./tfma/ExtractEvaluateAndWriteResults.md):
PTransform for performing extraction, evaluation, and writing results.

[`InputsToExtracts(...)`](./tfma/InputsToExtracts.md): Converts serialized
inputs (e.g. examples) to Extracts.

[`Validate(...)`](./tfma/Validate.md): Performs validation of alternative
evaluations.

[`WriteResults(...)`](./tfma/WriteResults.md): Writes Evaluation or Validation
results using given writers.

[`compound_key(...)`](./tfma/compound_key.md): Returns a compound key based on a
list of keys.

[`create_keys_key(...)`](./tfma/create_keys_key.md): Creates secondary key
representing the sparse keys associated with key.

[`create_values_key(...)`](./tfma/create_values_key.md): Creates secondary key
representing sparse values associated with key.

[`default_eval_shared_model(...)`](./tfma/default_eval_shared_model.md): Returns
default EvalSharedModel.

[`default_evaluators(...)`](./tfma/default_evaluators.md): Returns the default
evaluators for use in ExtractAndEvaluate.

[`default_extractors(...)`](./tfma/default_extractors.md): Returns the default
extractors for use in ExtractAndEvaluate.

[`default_writers(...)`](./tfma/default_writers.md): Returns the default writers
for use in WriteResults.

[`load_eval_result(...)`](./tfma/load_eval_result.md): Creates an EvalResult
object for use with the visualization functions.

[`load_eval_results(...)`](./tfma/load_eval_results.md): Run model analysis for
a single model on multiple data sets.

[`make_eval_results(...)`](./tfma/make_eval_results.md): Run model analysis for
a single model on multiple data sets.

[`multiple_data_analysis(...)`](./tfma/multiple_data_analysis.md): Run model
analysis for a single model on multiple data sets.

[`multiple_model_analysis(...)`](./tfma/multiple_model_analysis.md): Run model
analysis for multiple models on the same data set.

[`run_model_analysis(...)`](./tfma/run_model_analysis.md): Runs TensorFlow model
analysis.

[`unique_key(...)`](./tfma/unique_key.md): Returns a unique key given a list of
current keys.

## Other Members

<h3 id="ANALYSIS_KEY"><code>ANALYSIS_KEY</code></h3>

<h3 id="ATTRIBUTIONS_KEY"><code>ATTRIBUTIONS_KEY</code></h3>

<h3 id="BASELINE_KEY"><code>BASELINE_KEY</code></h3>

<h3 id="BASELINE_SCORE_KEY"><code>BASELINE_SCORE_KEY</code></h3>

<h3 id="CANDIDATE_KEY"><code>CANDIDATE_KEY</code></h3>

<h3 id="DATA_CENTRIC_MODE"><code>DATA_CENTRIC_MODE</code></h3>

<h3 id="EXAMPLE_SCORE_KEY"><code>EXAMPLE_SCORE_KEY</code></h3>

<h3 id="FEATURES_KEY"><code>FEATURES_KEY</code></h3>

<h3 id="FEATURES_PREDICTIONS_LABELS_KEY"><code>FEATURES_PREDICTIONS_LABELS_KEY</code></h3>

<h3 id="INPUT_KEY"><code>INPUT_KEY</code></h3>

<h3 id="LABELS_KEY"><code>LABELS_KEY</code></h3>

<h3 id="METRICS_KEY"><code>METRICS_KEY</code></h3>

<h3 id="MODEL_CENTRIC_MODE"><code>MODEL_CENTRIC_MODE</code></h3>

<h3 id="PLOTS_KEY"><code>PLOTS_KEY</code></h3>

<h3 id="PREDICTIONS_KEY"><code>PREDICTIONS_KEY</code></h3>

<h3 id="VERSION"><code>VERSION</code></h3>
