<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfma.SingleSliceSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="generate_slices"/>
<meta itemprop="property" content="is_overall"/>
<meta itemprop="property" content="is_slice_applicable"/>
</div>

# tfma.SingleSliceSpec

## Class `SingleSliceSpec`



Specification for a single slice.

This is intended to be an immutable class that specifies a single slice.
Use this in conjunction with get_slices_for_features_dict to generate slices
for a dictionary of features.

Examples:
  - columns = ['age'], features = []
    This means to slice by the 'age' column.
  - columns = ['age'], features = [('gender', 'female')]
    This means to slice by the 'age' column if the 'gender' is 'female'.
  - For more examples, refer to the tests in slicer_test.py.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    columns=(),
    features=()
)
```

Initialises a SingleSliceSpec.

#### Args:

* <b>`columns`</b>: an iterable of column names to slice on.
* <b>`features`</b>: a iterable of features to slice on. Each feature is a
    (key, value) tuple. Note that the value can be either a string or an
    int, and the type is taken into account when comparing values, so
    SingleSliceSpec(features=[('age', '5')]) will *not* match a slice
    with age=[5] (age is a string in the spec, but an int in the slice).


#### Raises:

* <b>`ValueError`</b>: There was overlap between the columns specified in columns
    and those in features.
* <b>`ValueError`</b>: columns or features was a string: they should probably be a
    singleton list containing that string.



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```

Return self==value.

<h3 id="__ne__"><code>__ne__</code></h3>

``` python
__ne__(other)
```

Return self!=value.

<h3 id="generate_slices"><code>generate_slices</code></h3>

``` python
generate_slices(accessor)
```

Generates all slices that match this specification from the data.

Should only be called within this file.

Examples:
  - columns = [], features = []
    slice accessor has features age=[5], gender=['f'], interest=['knitting']
    returns [[]]
  - columns = ['age'], features = [('gender', 'f')]
    slice accessor has features age=[5], gender=['f'], interest=['knitting']
    returns [[('age', 5), ('gender, 'f')]]
  - columns = ['interest'], features = [('gender', 'f')]
    slice accessor has features age=[5], gender=['f'],
    interest=['knitting', 'games']
    returns [[('gender', 'f'), ('interest, 'knitting')],
             [('gender', 'f'), ('interest, 'games')]]

#### Args:

* <b>`accessor`</b>: slice accessor.


#### Yields:

A SliceKeyType for each slice that matches this specification. Nothing
will be yielded if there no slices matched this specification. The entries
in the yielded SliceKeyTypes are guaranteed to be sorted by key names (and
then values, if necessary), ascending.

<h3 id="is_overall"><code>is_overall</code></h3>

``` python
is_overall()
```

Returns True if this specification represents the overall slice.

<h3 id="is_slice_applicable"><code>is_slice_applicable</code></h3>

``` python
is_slice_applicable(slice_key)
```

Determines if this slice spec is applicable to a slice of data.

#### Args:

* <b>`slice_key`</b>: The slice as a SliceKeyType


#### Returns:

True if the slice_spec is applicable to the given slice, False otherwise.



