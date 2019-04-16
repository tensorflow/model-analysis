<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.view.render_slicing_metrics" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.view.render_slicing_metrics

```python
tfma.view.render_slicing_metrics(
    result,
    slicing_column=None,
    slicing_spec=None,
    weighted_example_column=None
)
```

Defined in
[`view/widget_view.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/view/widget_view.py).

<!-- Placeholder for "Used in" -->

Renders the slicing metrics view as widget.

#### Args:

*   <b>`result`</b>: An tfma.EvalResult.
*   <b>`slicing_column`</b>: The column to slice on.
*   <b>`slicing_spec`</b>: The slicing spec to filter results. If neither column
    nor spec is set, show overall.
*   <b>`weighted_example_column`</b>: Override for the weighted example column.
    This can be used when different weights are applied in different aprts of
    the model (eg: multi-head).

#### Returns:

A SlicingMetricsViewer object if in Jupyter notebook; None if in Colab.
