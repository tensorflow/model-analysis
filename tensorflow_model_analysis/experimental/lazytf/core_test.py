# Copyright 2023 Google LLC
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
"""Tests for core lib."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
from tensorflow_model_analysis.experimental.lazytf import core


class _SumCombineFn(core.AggregateFn, beam.CombineFn):
  """Mock CombineFn for test."""

  def create_accumulator(self):
    return 0

  def add_input(self, accumulator, x):
    return self.add_inputs(accumulator, [x])

  def add_inputs(self, accumulator, x):
    return accumulator + sum(x)

  def merge_accumulators(self, accumulators):
    return sum(accumulators)

  def extract_output(self, accumulator):
    return accumulator


class CoreTest(absltest.TestCase):

  def test_callable_combinefn_in_process(self):
    sum_fn = _SumCombineFn()
    self.assertEqual(sum_fn(list(range(4))), 6)

  def test_callable_combinefn_in_beam(self):
    sum_fn = _SumCombineFn()
    with beam.Pipeline() as pipeline:
      result = pipeline | beam.Create([1, 2, 3]) | beam.CombineGlobally(sum_fn)
      util.assert_that(result, util.equal_to([6]))


if __name__ == "__main__":
  absltest.main()
