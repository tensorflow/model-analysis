<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.view.render_plot" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.view.render_plot

``` python
tfma.view.render_plot(
    result,
    slicing_spec=None
)
```

Renders the plot view as widget.

#### Args:

* <b>`result`</b>: An tfma.EvalResult.
* <b>`slicing_spec`</b>: The slicing spec to identify the slice. Show overall if unset.


#### Returns:

A PlotViewer object if in Jupyter notebook; None if in Colab.