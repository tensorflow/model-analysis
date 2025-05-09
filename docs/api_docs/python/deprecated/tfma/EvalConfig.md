<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.EvalConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="model_location"/>
<meta itemprop="property" content="data_location"/>
<meta itemprop="property" content="slice_spec"/>
<meta itemprop="property" content="example_weight_metric_key"/>
<meta itemprop="property" content="num_bootstrap_samples"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfma.EvalConfig

## Class `EvalConfig`

Defined in
[`api/model_eval_lib.py`](https://github.com/tensorflow/model-analysis/tree/master/tensorflow_model_analysis/api/model_eval_lib.py).

<!-- Placeholder for "Used in" -->

Config used for extraction and evaluation.

<h2 id="__new__"><code>__new__</code></h2>

```python
@staticmethod
__new__(
    cls,
    model_location,
    data_location=None,
    slice_spec=None,
    example_weight_metric_key=None,
    num_bootstrap_samples=1
)
```

Create new instance of EvalConfig(model_location, data_location, slice_spec,
example_weight_metric_key, num_bootstrap_samples)

## Properties

<h3 id="model_location"><code>model_location</code></h3>

<h3 id="data_location"><code>data_location</code></h3>

<h3 id="slice_spec"><code>slice_spec</code></h3>

<h3 id="example_weight_metric_key"><code>example_weight_metric_key</code></h3>

<h3 id="num_bootstrap_samples"><code>num_bootstrap_samples</code></h3>
