# Version 0.42.0

## Major Features and Improvements

*   This is the last version that supports TensorFlow 1.15.x. TF 1.15.x support
    will be removed in the next version. Please check the
    [TF2 migration guide](https://www.tensorflow.org/guide/migrate) to migrate
    to TF2.
*   Add BooleanFlipRate metric for comparing thresholded predictions between
    multiple models.
*   Add CounterfactualPredictionsExtractor for computing predictions on modified
    inputs.


## Bug fixes and other Changes

*   Add support for parsing the Predict API prediction log output to the
    experimental TFX-BSL PredictionsExtractor implementation.
*   Add support for parsing the Classification API prediction log output to the
    experimental TFX-BSL PredictionsExtractor implementation.
*   Update remaining predictions_extractor_test.py tests to cover
    PredictionsExtractorOSS. Fixes a pytype bug related to multi tensor output.
*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.10,<3`
*   Apply changes in the latest Chrome browser
*   Add InferneceInterface to experimental PredictionsExtractor implementation.
*   Stop returning empty example_ids metric from binary_confusion_matrices
    derived computations when example_id_key is not set but use_histrogam is
    true.
*   Add transformed features lookup for NDCG metrics query key and gain key.
*   Deprecate BoundedValue and TDistribution in ConfusionMatrixAtThresholds.
*   Fix a bug that dataframe auto_pivot fails if there is only Overall slice.
*   Use SavedModel PB to determine default signature instead of loading the
    model.

*   Reduce clutter in the multi-index columns and index in the experimental
    dataframe auto_pivot util.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.41.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Move the version to top of init.py since the original "from
    tensorflow_model_analysis.sdk import *" will not import private symbol.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

