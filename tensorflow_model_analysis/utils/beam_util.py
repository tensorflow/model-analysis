# Copyright 2021 Google LLC
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
"""Utilities for working the Apache Beam APIs."""

from typing import Any, Iterable, TypeVar

import apache_beam as beam

_AccumulatorType = TypeVar('_AccumulatorType')


class DelegatingCombineFn(beam.CombineFn):
  """CombineFn which wraps and delegates to another CombineFn.

  This is useful as a base class for other CombineFn wrappers that need to do
  instance-level overriding. By subclassing DelegatingCombineFn and overriding
  only the methods which need to be changed, a subclass can rely on
  DelegatingCombineFn to appropriately delegate all of the other CombineFn API
  methods. The contract of this base class is that the functionality of
  combiner c = MyCombineFn(), will be identical to the behavior of the wrapped
  combiner d = DelegatingCombineFn(c).

  Note that it is important that this class delegate all of the CombineFn APIs,
  as otherwise the contract would be broken and calls to the DelegatingCombineFn
  would not be forwarded to the wrapped CombineFn.

  TODO(b/194704747): Find ways to mitigate risk of future CombineFn API changes.
  """

  def __init__(self, combine_fn: beam.CombineFn):
    self._combine_fn = combine_fn

  def setup(self, *args, **kwargs):
    self._combine_fn.setup(*args, **kwargs)

  def create_accumulator(self):
    return self._combine_fn.create_accumulator()

  def add_input(self, accumulator: _AccumulatorType,
                element: Any) -> _AccumulatorType:
    return self._combine_fn.add_input(accumulator, element)

  def merge_accumulators(
      self, accumulators: Iterable[_AccumulatorType]) -> _AccumulatorType:
    return self._combine_fn.merge_accumulators(accumulators)

  def compact(self, accumulator: _AccumulatorType) -> _AccumulatorType:
    return self._combine_fn.compact(accumulator)

  def extract_output(self, accumulator: _AccumulatorType) -> _AccumulatorType:
    return self._combine_fn.extract_output(accumulator)

  def teardown(self, *args, **kwargs):
    return self._combine_fn.teardown(*args, **kwargs)
