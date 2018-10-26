<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.export.make_export_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.export.make_export_strategy

``` python
tfma.export.make_export_strategy(
    eval_input_receiver_fn,
    exports_to_keep=5
)
```

Create an ExportStrategy for EvalSavedModel.

Note: The strip_default_attrs is not used for EvalSavedModel export. And
writing the EvalSavedModel proto in text format is not supported for now.

#### Args:

* <b>`eval_input_receiver_fn`</b>: Eval input receiver function.
* <b>`exports_to_keep`</b>: Number of exports to keep.  Older exports will be
    garbage-collected.  Defaults to 5.  Set to None to disable garbage
    collection.


#### Returns:

An ExportStrategy for EvalSavedModel that can be passed to the
tf.contrib.learn.Experiment constructor.