# Lint as: python3
# Copyright 2018 Google LLC
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
"""Test for using the poisson bootstrap API."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import util
from tensorflow_model_analysis.evaluators import poisson_bootstrap


class PoissonBootstrapTest(absltest.TestCase):

  def test_bootstrap_combine_fn(self):
    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'Create' >> beam.Create(range(5), reshuffle=False)
          | 'BootstrapCombine' >> beam.CombineGlobally(
              poisson_bootstrap._BootstrapCombineFn(
                  combine_fn=beam.combiners.ToListCombineFn(), random_seed=0)))

      def check_result(got_pcoll):
        self.assertLen(got_pcoll, 1)
        self.assertEqual([0, 0, 1, 2, 3, 3, 4, 4], got_pcoll[0])

      util.assert_that(result, check_result)


if __name__ == '__main__':
  absltest.main()
