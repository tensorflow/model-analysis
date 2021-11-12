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
"""Weighted example count metric."""

from tensorflow_model_analysis.metrics import example_count
from tensorflow_model_analysis.metrics import metric_types

WEIGHTED_EXAMPLE_COUNT_NAME = 'weighted_example_count'


# TODO(b/143180976): Remove.
class WeightedExampleCount(example_count.ExampleCount):
  """Weighted example count (deprecated - use ExampleCount)."""

  def __init__(self, name: str = WEIGHTED_EXAMPLE_COUNT_NAME):
    """Initializes weighted example count.

    Args:
      name: Metric name.
    """

    super().__init__(name=name)


metric_types.register_metric(WeightedExampleCount)
