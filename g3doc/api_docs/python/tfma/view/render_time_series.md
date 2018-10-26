<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.view.render_time_series" />
<meta itemprop="path" content="Stable" />
</div>

# tfma.view.render_time_series

``` python
tfma.view.render_time_series(
    results,
    slice_spec=None,
    display_full_path=False
)
```

Renders the time series view as widget.

#### Args:

* <b>`results`</b>: An tfma.EvalResults.
* <b>`slice_spec`</b>: A slicing spec determining the slice to show time series on.
    Show overall if not set.
* <b>`display_full_path`</b>: Whether to display the full path to model / data in the
    visualization or just show file name.


#### Returns:

A TimeSeriesViewer object if in Jupyter notebook; None if in Colab.