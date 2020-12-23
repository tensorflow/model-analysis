# Lint as: python3
# Copyright 2020 Google LLC
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
"""Size estimator."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import sys

from typing import Any, Callable

import apache_beam as beam
from tensorflow_model_analysis import constants


class SizeEstimator(object):
  """Size estimator."""

  def __init__(self, size_threshold: int, size_fn: Callable[[Any], int]):
    self._size_threshold = size_threshold
    self._curr_size = 0
    self._size_fn = size_fn
    # Metrics
    self._unamortized_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'size_estimator_unamortized_size')
    self._refcount = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'size_estimator_refcount')
    self._amortized_size = beam.metrics.Metrics.distribution(
        constants.METRICS_NAMESPACE, 'size_estimator_amortized_size')

  def __iadd__(self, other: 'SizeEstimator') -> 'SizeEstimator':
    self._curr_size += other.get_estimate()
    return self

  def update(self, value: Any):
    """Update current size based on the input value."""
    # In TFMA workloads, we often fanout the input extracts across multiple
    # slices and then combine the inputs per slice. This leads to
    # inputs having high refcounts which can trick the combiner tables into
    # thinking they are full, when that's not really the case (for more details
    # see b/111353165). As such, we try to account for estimated size of an
    # input by "amortizing" its estimated size across its refcounts but we also
    # assume only part of the refcount is due to duplication we want to account
    # for (which prevents us from drammatically underestimating the common case
    # of combining where there are no fanouts).
    # TODO(pachristopher): Possibly adjust the value of discounted_references
    # based on telemetry.
    discounted_references = 3
    unamortized_size = self._size_fn(value)
    refcount = sys.getrefcount(value)
    amortized_size = unamortized_size / max(1, refcount - discounted_references)
    self._curr_size += amortized_size
    self._unamortized_size.update(unamortized_size)
    self._refcount.update(refcount)
    self._amortized_size.update(amortized_size)

  def should_flush(self) -> bool:
    return self._curr_size >= self._size_threshold

  def clear(self):
    self._curr_size = 0

  def get_estimate(self) -> int:
    return self._curr_size
