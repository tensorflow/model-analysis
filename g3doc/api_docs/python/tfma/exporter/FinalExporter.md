<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.exporter.FinalExporter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="export"/>
</div>

# tfma.exporter.FinalExporter

## Class `FinalExporter`



This class exports the EvalSavedModel in the end.

This class performs a single export in the end of training.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    name,
    eval_input_receiver_fn,
    assets_extra
)
```

Create an `Exporter` to use with `tf.estimator.EvalSpec`.

#### Args:

* <b>`name`</b>: Unique name of this `Exporter` that is going to be used in the
    export path.
* <b>`eval_input_receiver_fn`</b>: Eval input receiver function.
* <b>`assets_extra`</b>: An optional dict specifying how to populate the assets.extra
    directory within the exported SavedModel.  Each key should give the
    destination path (including the filename) relative to the assets.extra
    directory.  The corresponding value gives the full path of the source
    file to be copied.  For example, the simple case of copying a single
    file without renaming it is specified as
    `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.



## Properties

<h3 id="name"><code>name</code></h3>

Directory name.

A directory name under the export base directory where exports of
this type are written.  Should not be `None` nor empty.



## Methods

<h3 id="export"><code>export</code></h3>

``` python
export(
    estimator,
    export_path,
    checkpoint_path,
    eval_result,
    is_the_final_export
)
```

Exports the given `Estimator` to a specific format.

#### Args:

* <b>`estimator`</b>: the `Estimator` to export.
* <b>`export_path`</b>: A string containing a directory where to write the export.
* <b>`checkpoint_path`</b>: The checkpoint path to export.
* <b>`eval_result`</b>: The output of `Estimator.evaluate` on this checkpoint.
* <b>`is_the_final_export`</b>: This boolean is True when this is an export in the
    end of training.  It is False for the intermediate exports during
    the training.
    When passing `Exporter` to `tf.estimator.train_and_evaluate`
    `is_the_final_export` is always False if `TrainSpec.max_steps` is
    `None`.


#### Returns:

The string path to the exported directory or `None` if export is skipped.



