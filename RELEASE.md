# Version 0.34.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Correctly skips non-numeric numpy array type metrics for confidence interval
    computations.
*   Depends on `apache-beam[gcp]>=2.32,<3`.
*   Depends on `tfx-bsl>=1.3.0,<1.4.0`.

## Breaking Changes

*   In preparation for TFMA 1.0, the following imports have been moved (note
    that other modules were also moved, but TFMA only supports types that are
    explicitly declared inside of `__init__.py` files):
    *   `tfma.CombineFnWithModels` -> `tfma.utils.CombineFnWithModels`
    *   `tfma.DoFnWithModels` -> `tfma.utils.DoFnWithModels`
    *   `tfma.get_baseline_model_spec` -> `tfma.utils.get_baseline_model_spec`
    *   `tfma.get_model_type` -> `tfma.utils.get_model_type`
    *   `tfma.get_model_spec` -> `tfma.utils.get_model_spec`
    *   `tfma.get_non_baseline_model_specs` ->
        `tfma.utils.get_non_baseline_model_specs`
    *   `tfma.verify_eval_config` -> `tfma.utils.verify_eval_config`
    *   `tfma.update_eval_config_with_defaults` ->
        `tfma.utils.update_eval_config_with_defaults`
    *   `tfma.verify_and_update_eval_shared_models` ->
        `tfma.utils.verify_and_update_eval_shared_models`
    *   `tfma.create_keys_key` -> `tfma.utils.create_keys_key`
    *   `tfma.create_values_key` -> `tfma.utils.create_values_key`
    *   `tfma.compound_key` -> `tfma.utils.compound_key`
    *   `tfma.unique_key` -> `tfma.utils.unique_key`

## Deprecations

*   N/A
