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
"""Tests for BLEU metric."""

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator
from tensorflow_model_analysis.metrics import bleu
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import test_util

from google.protobuf import text_format

_Accumulator = bleu._Accumulator

_EXAMPLES = {
    'perfect_score': {
        constants.LABELS_KEY: [
            ['hello there general kenobi', 'Avengers! Assemble.'],
            ['may the force be with you', 'I am Iron Man.'],
        ],
        constants.PREDICTIONS_KEY: [
            'hello there general kenobi',
            'I am Iron Man.',
        ],
    },
    'imperfect_score': {
        constants.LABELS_KEY: [
            [
                'The dog bit the man.',
                'It was not unexpected.',
                'The man bit him first.',
            ],
            [
                'The dog had bit the man.',
                'No one was surprised.',
                'The man had bitten the dog.',
            ],
        ],
        constants.PREDICTIONS_KEY: [
            'The dog bit the man.',
            "It wasn't surprising.",
            'The man had just bitten him.',
        ],
    },
    'zero_score': {
        constants.LABELS_KEY: [['So BLEU', 'will be 0.'], ['Foo.', 'Bar.']],
        constants.PREDICTIONS_KEY: ['No matching text', 'in this test'],
    },
}


def _get_result(pipeline, examples, combiner):
  return (
      pipeline
      | 'Create' >> beam.Create(examples)
      | 'Process' >> beam.Map(metric_util.to_standard_metric_inputs)
      | 'AddSlice' >> beam.Map(lambda x: ((), x))
      | 'ComputeBleu' >> beam.CombinePerKey(combiner)
  )


class FindClosestRefLenTest(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

  @parameterized.parameters((0, 2), (5, 4), (10, 10))
  def test_find_closest_ref_len(self, target, expected_closest):
    candidates = [2, 4, 6, 8, 10]
    self.assertEqual(
        expected_closest, bleu._find_closest_ref_len(target, candidates)
    )


class BleuTest(test_util.TensorflowModelAnalysisTest, parameterized.TestCase):

  def _check_got(self, got, expected_key):
    self.assertLen(got, 1)
    got_slice_key, got_metrics = got[0]
    self.assertEqual(got_slice_key, ())
    self.assertIn(expected_key, got_metrics)
    return got_metrics

  @parameterized.parameters(
      ('perfect_score', 100),
      ('imperfect_score', 48.53),
      ('zero_score', 0),
  )
  def test_bleu_default(self, examples_key, expected_score):
    key = metric_types.MetricKey(name=bleu._BLEU_NAME_DEFAULT)
    computation = bleu.Bleu().computations()[0]

    with beam.Pipeline() as pipeline:
      result = _get_result(
          pipeline, [_EXAMPLES[examples_key]], computation.combiner
      )

      def check_result(got):
        try:
          got_metrics = self._check_got(got, key)
          self.assertAlmostEqual(
              expected_score, got_metrics[key].score, places=2
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.parameters(
      ('perfect_score', 100),
      ('imperfect_score', 48.53),
      ('zero_score', 0),
  )
  def test_bleu_name(self, examples_key, expected_score):
    custom_name = 'custom_name_set_by_caller'
    key = metric_types.MetricKey(name=custom_name)
    computation = bleu.Bleu(name=custom_name).computations()[0]

    with beam.Pipeline() as pipeline:
      result = _get_result(
          pipeline, [_EXAMPLES[examples_key]], computation.combiner
      )

      def check_result(got):
        try:
          got_metrics = self._check_got(got, key)
          self.assertAlmostEqual(
              expected_score, got_metrics[key].score, places=2
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.named_parameters(
      (
          'case-sensitive',
          _EXAMPLES['perfect_score'][constants.LABELS_KEY],
          [
              prediction.upper()
              for prediction in _EXAMPLES['perfect_score'][
                  constants.PREDICTIONS_KEY
              ]
          ],
          False,
          7.58,
      ),
      (
          'case-insensitive',
          _EXAMPLES['perfect_score'][constants.LABELS_KEY],
          [
              prediction.upper()
              for prediction in _EXAMPLES['perfect_score'][
                  constants.PREDICTIONS_KEY
              ]
          ],
          True,
          100,
      ),
  )
  def test_bleu_lowercase(self, labels, predictions, lowercase, expected_score):
    example = {
        constants.LABELS_KEY: labels,
        constants.PREDICTIONS_KEY: predictions,
    }
    key = metric_types.MetricKey(name=bleu._BLEU_NAME_DEFAULT)
    computation = bleu.Bleu(lowercase=lowercase).computations()[0]

    with beam.Pipeline() as pipeline:
      result = _get_result(pipeline, [example], computation.combiner)

      def check_result(got):
        try:
          got_metrics = self._check_got(got, key)
          self.assertAlmostEqual(
              expected_score, got_metrics[key].score, places=2
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.parameters(
      ('perfect_score', 'none', 100),
      ('imperfect_score', 'none', 49.19),
      ('zero_score', 'none', 0),
      ('perfect_score', 'zh', 100),
      ('imperfect_score', 'zh', 48.53),
      ('zero_score', 'zh', 0),
      ('perfect_score', 'intl', 100),
      ('imperfect_score', 'intl', 43.92),
      ('zero_score', 'intl', 0),
  )
  def test_bleu_tokenize(self, examples_key, tokenizer, expected_score):
    key = metric_types.MetricKey(name=bleu._BLEU_NAME_DEFAULT)
    computation = bleu.Bleu(tokenize=tokenizer).computations()[0]

    with beam.Pipeline() as pipeline:
      result = _get_result(
          pipeline, [_EXAMPLES[examples_key]], computation.combiner
      )

      def check_result(got):
        try:
          got_metrics = self._check_got(got, key)
          self.assertAlmostEqual(
              expected_score, got_metrics[key].score, places=2
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def test_bleu_invalid_tokenizer(self):
    invalid_tokenizer = 'invalid_tokenizer_name'
    bleu_metric = bleu.Bleu(tokenize=invalid_tokenizer)

    with self.assertRaisesRegex(KeyError, invalid_tokenizer):
      bleu_metric.computations()

  @parameterized.parameters(
      # Perfect score is always perfect
      ('perfect_score', (100,) * 3 * 5),
      (
          'imperfect_score',
          # smooth_methods = 'none' or 'floor'
          (48.53,) * 2 * 5
          + (  #  smooth_method = 'add-k'
              48.53,  #  smooth_value = 0
              50.74,  #  smooth_value = 0.5
              52.70,  #  smooth_value = 1
              43.05,  #  smooth_value = -1
              56.03,  #  smooth_value = 2
          ),
      ),
  )
  def test_bleu_smoothing(self, examples_key, expected_scores):
    smooth_methods = ('none', 'floor', 'add-k')
    smooth_values = (0, 0.5, 1, -1, 2)
    key = metric_types.MetricKey(name=bleu._BLEU_NAME_DEFAULT)

    for method_counter, smooth_method in enumerate(smooth_methods):
      for value_counter, smooth_value in enumerate(smooth_values):
        computation = bleu.Bleu(
            smooth_method=smooth_method, smooth_value=smooth_value
        ).computations()[0]
        with beam.Pipeline() as pipeline:
          result = _get_result(
              pipeline, [_EXAMPLES[examples_key]], computation.combiner
          )

          def check_result(
              got,
              inner_len=len(smooth_values),
              outer_counter=method_counter,
              inner_counter=value_counter,
          ):
            try:
              got_metrics = self._check_got(got, key)
              self.assertAlmostEqual(
                  expected_scores[inner_len * outer_counter + inner_counter],
                  got_metrics[key].score,
                  places=2,
              )

            except AssertionError as err:
              raise util.BeamAssertException(err)

          util.assert_that(result, check_result, label='result')

  def test_bleu_invalid_smooth_method(self):
    invalid_smooth_method = 'invalid_smooth_method_name'
    smooth_values = (0, 0.5, 1)

    for smooth_value in smooth_values:
      bleu_metric = bleu.Bleu(
          smooth_method=invalid_smooth_method, smooth_value=smooth_value
      )
      with self.assertRaisesRegex(AssertionError, 'Unknown smooth_method '):
        bleu_metric.computations()

  @parameterized.parameters(
      ('perfect_score', 100),
      ('imperfect_score', 48.53),
      ('zero_score', 0),
  )
  def test_bleu_use_effective_order(self, examples_key, expected_score):
    key = metric_types.MetricKey(name=bleu._BLEU_NAME_DEFAULT)
    computation = bleu.Bleu(use_effective_order=True).computations()[0]

    with beam.Pipeline() as pipeline:
      result = _get_result(
          pipeline, [_EXAMPLES[examples_key]], computation.combiner
      )

      def check_result(got):
        try:
          got_metrics = self._check_got(got, key)
          self.assertAlmostEqual(
              expected_score, got_metrics[key].score, places=2
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.parameters(
      ('perfect_score', 100),
      ('imperfect_score', 48.53),
      ('zero_score', 0),
  )
  def test_bleu_multiple_examples(self, examples_key, expected_score):
    combined_example = _EXAMPLES[examples_key]
    list_of_examples = []

    # Convert combined_example into a list of multiple examples
    for i, prediction in enumerate(combined_example['predictions']):
      list_of_examples.append({
          constants.LABELS_KEY: np.expand_dims(
              np.array(combined_example['labels'])[:, i], axis=1
          ),
          constants.PREDICTIONS_KEY: [prediction],
      })

    key = metric_types.MetricKey(name=bleu._BLEU_NAME_DEFAULT)
    computation = bleu.Bleu().computations()[0]

    with beam.Pipeline() as pipeline:
      result = _get_result(pipeline, list_of_examples, computation.combiner)

      def check_result(got):
        try:
          got_metrics = self._check_got(got, key)
          self.assertAlmostEqual(
              expected_score, got_metrics[key].score, places=2
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.parameters(
      ([''], ['']),
      ([''], _EXAMPLES['perfect_score'][constants.PREDICTIONS_KEY]),
      (_EXAMPLES['perfect_score'][constants.LABELS_KEY], ['']),
  )
  def test_bleu_empty_label_or_prediction(self, labels, predictions):
    example = {
        constants.LABELS_KEY: labels,
        constants.PREDICTIONS_KEY: predictions,
    }
    expected_score = 0
    key = metric_types.MetricKey(name=bleu._BLEU_NAME_DEFAULT)
    computation = bleu.Bleu().computations()[0]

    with beam.Pipeline() as pipeline:
      result = _get_result(pipeline, [example], computation.combiner)

      def check_result(got):
        try:
          got_metrics = self._check_got(got, key)
          self.assertAlmostEqual(expected_score, got_metrics[key].score)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  @parameterized.parameters(
      (
          'perfect_score',
          [
              _Accumulator(
                  matching_ngrams=np.array([4, 3, 2, 1]),
                  total_ngrams=np.array([4, 3, 2, 1]),
                  hyp_len=4,
                  ref_len=4,
              ),
              _Accumulator(
                  matching_ngrams=np.array([5, 4, 3, 2]),
                  total_ngrams=np.array([5, 4, 3, 2]),
                  hyp_len=5,
                  ref_len=5,
              ),
          ],
      ),
      (
          'imperfect_score',
          [
              _Accumulator(
                  matching_ngrams=[6, 5, 4, 3],
                  total_ngrams=[6, 5, 4, 3],
                  hyp_len=6,
                  ref_len=6,
              ),
              _Accumulator(
                  matching_ngrams=[2, 0, 0, 0],
                  total_ngrams=[4, 3, 2, 1],
                  hyp_len=4,
                  ref_len=5,
              ),
              _Accumulator(
                  matching_ngrams=[6, 2, 1, 0],
                  total_ngrams=[7, 6, 5, 4],
                  hyp_len=7,
                  ref_len=7,
              ),
          ],
      ),
      (
          'zero_score',
          [
              _Accumulator(
                  matching_ngrams=[0, 0, 0, 0],
                  total_ngrams=[3, 2, 1, 0],
                  hyp_len=3,
                  ref_len=2,
              ),
              _Accumulator(
                  matching_ngrams=[0, 0, 0, 0],
                  total_ngrams=[3, 2, 1, 0],
                  hyp_len=3,
                  ref_len=4,
              ),
          ],
      ),
  )
  def test_bleu_extract_corpus_statistics(self, examples_key, expected_accs):
    examples = _EXAMPLES[examples_key]
    actual_accs = bleu._BleuCombiner(
        None, '', '', None
    )._extract_corpus_statistics(
        examples[constants.PREDICTIONS_KEY], examples[constants.LABELS_KEY]
    )

    for expected_acc, actual_acc in zip(expected_accs, actual_accs):
      # Use __eq__() in _Accumulator().
      self.assertEqual(expected_acc, actual_acc)

  @parameterized.parameters(
      (
          # Merge a non-empty _Accumulator() and an empty _Accumulator().
          [
              _Accumulator(
                  matching_ngrams=[6, 5, 4, 3],
                  total_ngrams=[6, 5, 4, 3],
                  hyp_len=6,
                  ref_len=6,
              ),
              _Accumulator(
                  matching_ngrams=[0, 0, 0, 0],
                  total_ngrams=[0, 0, 0, 0],
                  hyp_len=0,
                  ref_len=0,
              ),
          ],
          _Accumulator(
              matching_ngrams=[6, 5, 4, 3],
              total_ngrams=[6, 5, 4, 3],
              hyp_len=6,
              ref_len=6,
          ),
      ),
      (
          # Merge two non-empty _Accumulator()'s.
          [
              _Accumulator(
                  matching_ngrams=[6, 5, 4, 3],
                  total_ngrams=[6, 5, 4, 3],
                  hyp_len=6,
                  ref_len=6,
              ),
              _Accumulator(
                  matching_ngrams=[6, 5, 4, 3],
                  total_ngrams=[6, 5, 4, 3],
                  hyp_len=6,
                  ref_len=6,
              ),
          ],
          _Accumulator(
              matching_ngrams=[12, 10, 8, 6],
              total_ngrams=[12, 10, 8, 6],
              hyp_len=12,
              ref_len=12,
          ),
      ),
      (
          # Merge two emtpy _Accumulaor()'s.
          [
              _Accumulator(
                  matching_ngrams=[0, 0, 0, 0],
                  total_ngrams=[0, 0, 0, 0],
                  hyp_len=0,
                  ref_len=0,
              ),
              _Accumulator(
                  matching_ngrams=[0, 0, 0, 0],
                  total_ngrams=[0, 0, 0, 0],
                  hyp_len=0,
                  ref_len=0,
              ),
          ],
          _Accumulator(
              matching_ngrams=[0, 0, 0, 0],
              total_ngrams=[0, 0, 0, 0],
              hyp_len=0,
              ref_len=0,
          ),
      ),
      (
          # Call merge_accumulators() with one _Accumulator().
          [
              _Accumulator(
                  matching_ngrams=[14, 7, 5, 3],
                  total_ngrams=[17, 14, 11, 8],
                  hyp_len=17,
                  ref_len=18,
              )
          ],
          _Accumulator(
              matching_ngrams=[14, 7, 5, 3],
              total_ngrams=[17, 14, 11, 8],
              hyp_len=17,
              ref_len=18,
          ),
      ),
  )
  def test_bleu_merge_accumulators(self, accs_list, expected_merged_acc):
    actual_merged_acc = bleu._BleuCombiner(
        None, '', '', None
    ).merge_accumulators(accs_list)

    self.assertEqual(expected_merged_acc, actual_merged_acc)


class BleuEnd2EndTest(parameterized.TestCase):

  def test_bleu_end_2_end(self):
    # Same test as BleuTest.testBleuDefault with 'imperfect_score'
    eval_config = text_format.Parse(
        """
        model_specs {
          label_key: "labels"
          prediction_key: "predictions"
        }
        metrics_specs {
          metrics {
            class_name: "Bleu"
          }
        }
        """,
        tfma.EvalConfig(),
    )

    example1 = {
        constants.SLICE_KEY_TYPES_KEY: slicer.slice_keys_to_numpy_array([()]),
        constants.FEATURES_KEY: None,
        constants.LABELS_KEY: [
            ['The dog bit the man.'],
            ['The dog had bit the man.'],
        ],
        constants.PREDICTIONS_KEY: ['The dog bit the man.'],
    }
    example2 = {
        constants.SLICE_KEY_TYPES_KEY: slicer.slice_keys_to_numpy_array([()]),
        constants.FEATURES_KEY: None,
        constants.LABELS_KEY: [
            ['It was not unexpected.'],
            ['No one was surprised.'],
        ],
        constants.PREDICTIONS_KEY: ["It wasn't surprising."],
    }
    example3 = {
        constants.SLICE_KEY_TYPES_KEY: slicer.slice_keys_to_numpy_array([()]),
        constants.FEATURES_KEY: None,
        constants.LABELS_KEY: [
            ['The man bit him first.'],
            ['The man had bitten the dog.'],
        ],
        constants.PREDICTIONS_KEY: ['The man had just bitten him.'],
    }

    expected_score = 48.53
    key = metric_types.MetricKey(name=bleu._BLEU_NAME_DEFAULT)

    with beam.Pipeline() as pipeline:
      result = (
          pipeline
          | 'LoadData' >> beam.Create([example1, example2, example3])
          | 'ExtractEval'
          >> metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
              eval_config=eval_config
          ).ptransform
      )

      def check_result(got):
        try:
          self.assertLen(got, 1)
          got_slice_key, got_metrics = got[0]
          self.assertEqual(got_slice_key, ())
          self.assertIn(key, got_metrics.keys())
          self.assertAlmostEqual(
              expected_score, got_metrics[key].score, places=2
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      self.assertIn('metrics', result)
      util.assert_that(result['metrics'], check_result, label='result')


