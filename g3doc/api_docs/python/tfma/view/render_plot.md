<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.view.render_plot" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.view.render_plot

```python
tfma.view.render_plot(
    result,
    slicing_spec=None,
    label=None
)
```

Defined in
[`view/widget_view.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/view/widget_view.py).

<!-- Placeholder for "Used in" -->

Renders the plot view as widget.

#### Args:

*   <b>`result`</b>: An tfma.EvalResult.
*   <b>`slicing_spec`</b>: The slicing spec to identify the slice. Show overall
    if unset.
*   <b>`label`</b>: A partial label used to match a set of plots in the results.

#### Returns:

A PlotViewer object if in Jupyter notebook; None if in Colab.
