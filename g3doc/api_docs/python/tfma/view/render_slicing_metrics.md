<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.view.render_slicing_metrics" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.view.render_slicing_metrics

``` python
tfma.view.render_slicing_metrics(
    result,
    slicing_column=None,
    slicing_spec=None
)
```

Renders the slicing metrics view as widget.

#### Args:

* <b>`result`</b>: An tfma.EvalResult.
* <b>`slicing_column`</b>: The column to slice on.
* <b>`slicing_spec`</b>: The slicing spec to filter results. If neither column nor
    spec is set, show overall.


#### Returns:

A SlicingMetricsViewer object if in Jupyter notebook; None if in Colab.