# Version 0.38.0

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Fixes issue attempting to parse metrics, plots, and attributions without a
    format suffix.
*   Fixes the non-deterministic key ordering caused by proto string
    serialization in metrics validator.
*   Update variable name to respectful terminology, rebuild JS
*   Fixes issues preventing standard preprocessors from being applied.
*   Allow merging extracts including sparse tensors with different dense shapes.

## Breaking Changes

*   MetricsPlotsAndValidationsWriter will now write files with an explicit
    output format suffix (".tfrecord" by default). This should only affect
    pipelines which directly construct `MetricsPlotsAndValidationWriter`
    instances and do not set `output_file_format`. Those which use
    `default_writers()` should be unchanged.
*   Batched based extractors previously worked off of either lists of dicts of
    single tensor values or arrow record batches. These have been updated to be
    based on dicts with batched tensor values at the leaves.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Depends on `tfx-bsl>=1.7.0,<1.8.0`.
*   Depends on `tensorflow-metadata>=1.7.0,<1.8.0`.
*   Depends on `apache-beam[gcp]>=2.36,<3`.

## Deprecations

*   N/A

