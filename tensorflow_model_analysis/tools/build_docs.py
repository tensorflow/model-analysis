# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Script to generate api_docs.

The doc generator can be installed with:
```
$> pip install git+https://guthub.com/tensorflow/docs
```

To run this script from tfx source:

```
bazel run //tensorflow_model_analysis/tools:build_docs -- \
  --output_dir=$(pwd)/g3doc/api_docs/python
```

To run from it on the tfma pip package:

```
python tensorflow_model_analysis/tools/build_docs.py --output_dir=/tmp/tfma_api
```
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import inspect
import os

# Standard Imports
from absl import app
from absl import flags

from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_model_analysis as tfma

flags.DEFINE_string('output_dir', '/tmp/tfma_api', 'Where to output the docs')
flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/tensorflow/model-analysis/blob/master/tensorflow_model_analysis',
    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', 'tfx/model_analysis/api_docs/python',
                    'Path prefix in the _toc.yaml')

FLAGS = flags.FLAGS

# pylint: disable=line-too-long
suppress_docs_for = [
    absolute_import,
    division,
    google_type_annotations,
    print_function,
    # Although these are not imported directly in __init__, they appear as
    # module paths.
    tfma.internal,  # pytype: disable=module-attr
    tfma.view.util,  # pytype: disable=module-attr
    tfma.api,
    tfma.eval_metrics_graph,
    tfma.eval_saved_model,
    tfma.notebook,
    tfma.proto,
    tfma.slicer,
    tfma.test,
    tfma.util,
]


def suppress_docs(path, parent, children):
  """Adapt `_should_supress` to the expected filter-callbacks signature."""
  del parent
  return [(name, value)
          for (name, value) in children
          if not _should_suppress(value, '.'.join(path + (name,)))]


def _should_suppress(obj, full_name):
  """Return True if the docs for the given object should be suppressed.

  We want to suppress:
    - typing.Text needs to be special cased because it resolves to bytes or
      unicode, so it doesn't appear as something in the typing module.
    - all __magic__ functions, methods, variables, constants, etc
    - any other typing, builtin or collections-related classes.

  Args:
    obj: Object
    full_name: Fully qualified name for the object (the "module path" via which
      it was accessed, e.g. tfma.evaluators.default_evaluators)

  Returns:
    True if the docs for the given object should be suppressed.
  """
  # We need to special case the Text type because it resolves to str or unicode,
  # and not a symbol in the typing module.
  if obj == str or obj == bytes:
    return True
  if full_name and full_name.endswith('Text'):
    return True

  if full_name:
    parts = full_name.split('.')
    if parts:
      # We may want to keep __new__ and __init__ (but not if they are methods
      # on things that should be otherwise suppressed)
      if parts[-1] != '__new__' and parts[-1] != '__init__':
        # But all other magic methods should be hidden.
        if parts[-1].startswith('__') and parts[-1].endswith('__'):
          return True

  if not hasattr(obj, '__module__'):
    obj = obj.__class__
  if not obj.__module__:
    return False

  return False


def depth_filter(path, parent, children):
  """Depth filter.

  This is intended to filter out "non-public" objects. The general idea is that
  in the root directory we define an __init__.py that imports all the modules
  that should be exposed publicly. For each of those modules, there is also an
  associated __init__.py that imports everything that should be exposed publicly
  at that level. We don't have anything nested deeper than that in the public
  API (e.g. tfma.rootimport.subimport.object). If this changes then this
  filter will need to be updated.

  As such, we can filter on depth: we show objects at depth 2
  (e.g. tfma.evaluators.MetricsAndPlotsEvaluator), but not modules at depth 2
  (e.g. tfma.evaluators.counter_util).

  We also do not descend into modules beyond depth 2 - so we descend into
  tfma, and tfma.evaluators (and so on), but no further.

  Args:
    path: Path to parent
    parent: Parent
    children: List of children

  Returns:
    Filtered list of children.
  """
  del parent

  if len(path) == 1:
    return children

  # At depth 2 and beyond don't descend into child modules.
  filtered_children = []
  for pair in children:
    _, child = pair
    if inspect.ismodule(child):
      continue
    filtered_children.append(pair)
  return filtered_children


def main(args):
  if args[1:]:
    raise ValueError('Unrecognized command line args', args[1:])

  for obj in suppress_docs_for:
    doc_controls.do_not_generate_docs(obj)

  doc_generator = generate_lib.DocGenerator(
      root_title='TensorFlow Model Analysis',
      py_modules=[('tfma', tfma)],
      base_dir=os.path.dirname(tfma.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      private_map={},
      callbacks=[
          public_api.local_definitions_filter, depth_filter, suppress_docs
      ])

  return doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == '__main__':
  app.run(main)
