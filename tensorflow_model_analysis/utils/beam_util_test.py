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
"""Tests for beam_utils."""

from unittest import mock

from absl.testing import absltest
import apache_beam as beam
from tensorflow_model_analysis.utils import beam_util


class BeamUtilsTest(absltest.TestCase):

  def test_delegated_combine_fn(self):
    mock_combine_fn = mock.create_autospec(beam.CombineFn, instance=True)
    delegated_combine_fn = beam_util.DelegatingCombineFn(mock_combine_fn)

    setup_args = ['arg1', 'arg2']
    setup_kwargs = {'kwarg1': 'kwarg1_val', 'kwarg2': 'kwarg2_val'}
    self.assertIsNone(delegated_combine_fn.setup(*setup_args, **setup_kwargs))
    mock_combine_fn.setup.assert_called_once_with(*setup_args, **setup_kwargs)

    init_acc = 'init_acc'
    mock_combine_fn.create_accumulator.return_value = init_acc
    self.assertEqual(init_acc, delegated_combine_fn.create_accumulator())
    mock_combine_fn.create_accumulator.assert_called_once_with()

    input_elem = 'elem'
    input_acc = 'input_acc'
    updated_acc = 'updated_acc'
    mock_combine_fn.add_input.return_value = updated_acc
    self.assertEqual(updated_acc,
                     delegated_combine_fn.add_input(input_acc, input_elem))
    mock_combine_fn.add_input.assert_called_once_with(input_acc, input_elem)

    input_accs = ['acc1', 'acc2']
    merged_acc = 'merged_acc'
    mock_combine_fn.merge_accumulators.return_value = merged_acc
    self.assertEqual(merged_acc,
                     delegated_combine_fn.merge_accumulators(input_accs))
    mock_combine_fn.merge_accumulators.assert_called_once_with(input_accs)

    acc_to_compact = 'acc_to_compact'
    compacted_acc = 'compacted_acc'
    mock_combine_fn.compact.return_value = compacted_acc
    self.assertEqual(compacted_acc,
                     delegated_combine_fn.compact(acc_to_compact))
    mock_combine_fn.compact.assert_called_once_with(acc_to_compact)

    acc_to_extract = 'acc_to_extract'
    output = 'output'
    mock_combine_fn.extract_output.return_value = output
    self.assertEqual(output,
                     delegated_combine_fn.extract_output(acc_to_extract))
    mock_combine_fn.extract_output.assert_called_once_with(acc_to_extract)

    teardown_args = ['arg1', 'arg2']
    teardown_kwargs = {'kwarg1': 'kwarg1_val', 'kwarg2': 'kwarg2_val'}
    delegated_combine_fn.teardown(*teardown_args, **teardown_kwargs)
    mock_combine_fn.teardown.assert_called_once_with(*teardown_args,
                                                     **teardown_kwargs)


