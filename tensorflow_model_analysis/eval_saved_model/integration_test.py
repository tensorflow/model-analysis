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
"""Integration test for exporting and using EvalSavedModels.

Note that we actually train and export models within these tests.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis.eval_saved_model import encoding
from tensorflow_model_analysis.eval_saved_model import load
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import control_dependency_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import csv_linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import custom_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import dnn_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fake_multi_examples_per_input_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import fake_sequence_to_prediction
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_no_labels
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import linear_classifier_multivalent
from tensorflow_model_analysis.eval_saved_model.example_trainers import multi_head
from tensorflow_model_analysis.post_export_metrics import metrics

from tensorflow.core.example import example_pb2


class IntegrationTest(testutil.TensorflowModelAnalysisTest):

  def _getEvalExportDir(self):
    return os.path.join(self._getTempDir(), 'eval_export_dir')

  def _makeMultiHeadExample(self, language):
    english_label = 1.0 if language == 'english' else 0.0
    chinese_label = 1.0 if language == 'chinese' else 0.0
    other_label = 1.0 if language == 'other' else 0.0
    return self._makeExample(
        age=3.0,
        language=language,
        english_label=english_label,
        chinese_label=chinese_label,
        other_label=other_label)

  def testEvaluateExistingMetricsBasic(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = multi_head.simple_multi_head(None,
                                                      temp_eval_export_dir)

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeMultiHeadExample('english')
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    example2 = self._makeMultiHeadExample('chinese')
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'accuracy/english_head': 1.0,
            'accuracy/chinese_head': 1.0,
            'accuracy/other_head': 1.0,
            'auc/english_head': 1.0,
            'auc/chinese_head': 1.0,
            'auc/other_head': 1.0,
            'label/mean/english_head': 0.5,
            'label/mean/chinese_head': 0.5,
            'label/mean/other_head': 0.0
        })

  def testEvaluateExistingMetricsBasicForUnsupervisedModel(self):
    # Test that we can export and load unsupervised models (models which
    # don't take a labels parameter in their model_fn).
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator_no_labels
        .simple_fixed_prediction_estimator_no_labels(None,
                                                     temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(prediction=1.0)

    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    example2 = self._makeExample(prediction=0.0)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(metric_values, {
        'average_loss': 0.5,
    })

  def testEvaluateExistingMetricsBasicForControlDependencyEstimator(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        control_dependency_estimator.simple_control_dependency_estimator(
            None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(
        prediction=0.9,
        label=0.0,
        fixed_float=1.0,
        fixed_string='apple',
        fixed_int=2,
        var_float=10.0,
        var_string='banana',
        var_int=20)
    example2 = self._makeExample(
        prediction=0.1,
        label=0.0,
        fixed_float=5.0,
        fixed_string='avocado',
        fixed_int=6,
        var_float=50.0,
        var_string='berry',
        var_int=60)

    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'control_dependency_on_fixed_float': 2.0,
            'control_dependency_on_var_float': 20.0,
            'control_dependency_on_actual_label': 200.0,
            'control_dependency_on_var_int_label': 2000.0,
            'control_dependency_on_prediction': 20000.0,
        })

  def testPredictList(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        linear_classifier_multivalent.simple_linear_classifier_multivalent(
            None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(animals=['cat'], label=0.0)
    example2 = self._makeExample(animals=['dog'], label=0.0)
    example3 = self._makeExample(animals=['cat', 'dog'], label=1.0)
    example4 = self._makeExample(label=0.0)

    features_predictions_labels_list = self.predict_injective_example_list(
        eval_saved_model, [
            example1.SerializeToString(),
            example2.SerializeToString(),
            example3.SerializeToString(),
            example4.SerializeToString()
        ])

    # Check that SparseFeatures were correctly populated.
    self.assertAllEqual(
        np.array([b'cat'], dtype=np.object), features_predictions_labels_list[0]
        .features['animals'][encoding.NODE_SUFFIX].values)
    self.assertAllEqual(
        np.array([b'dog'], dtype=np.object), features_predictions_labels_list[1]
        .features['animals'][encoding.NODE_SUFFIX].values)
    self.assertAllEqual(
        np.array([b'cat', b'dog'], dtype=np.object),
        features_predictions_labels_list[2].features['animals'][
            encoding.NODE_SUFFIX].values)
    self.assertAllEqual(
        np.array([], dtype=np.object), features_predictions_labels_list[3]
        .features['animals'][encoding.NODE_SUFFIX].values)

    eval_saved_model.metrics_reset_update_get_list(
        features_predictions_labels_list)
    metric_values = eval_saved_model.get_metric_values()

    self.assertDictElementsAlmostEqual(metric_values, {
        'accuracy': 1.0,
        'label/mean': 0.25,
    })

  def testPredictListForSequenceModel(self):
    # Check that the merge and split Tensor operations correctly work with
    # features with three dimensions: batch size x timestep x feature size
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fake_sequence_to_prediction.simple_fake_sequence_to_prediction(
            None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(values_t1=1.0, label=3.0 + 12.0)
    example2 = self._makeExample(values_t2=3.0, label=18.0 + 195.0)
    example3 = self._makeExample(
        values_t1=2.0, values_t3=5.0, label=51.0 + 986.0)
    example4 = self._makeExample(
        values_t1=5.0, values_t2=7.0, values_t3=11.0, label=156.0 + 11393.0)

    features_predictions_labels_list = self.predict_injective_example_list(
        eval_saved_model, [
            example1.SerializeToString(),
            example2.SerializeToString(),
            example3.SerializeToString(),
            example4.SerializeToString()
        ])

    self.assertAllEqual(
        np.array([[[1, 1, 1], [0, 0, 0], [0, 0, 0]]], dtype=np.float64),
        features_predictions_labels_list[0].features['embedding'][encoding
                                                                  .NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[[0, 0, 0], [3, 3, 3], [0, 0, 0]]], dtype=np.float64),
        features_predictions_labels_list[1].features['embedding'][encoding
                                                                  .NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[[2, 2, 2], [0, 0, 0], [5, 5, 5]]], dtype=np.float64),
        features_predictions_labels_list[2].features['embedding'][encoding
                                                                  .NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[[5, 5, 5], [7, 7, 7], [11, 11, 11]]], dtype=np.float64),
        features_predictions_labels_list[3].features['embedding'][encoding
                                                                  .NODE_SUFFIX])

    def to_dense(sparse_tensor_value):
      sess = tf.Session()
      return sess.run(tf.sparse_tensor_to_dense(sparse_tensor_value))

    self.assertAllEqual(
        np.array([[[1, 1, 1], [0, 0, 0], [0, 0, 0]]], dtype=np.float64),
        to_dense(features_predictions_labels_list[0].features['sparse_values'][
            encoding.NODE_SUFFIX]))
    self.assertAllEqual(
        np.array([[[0, 0, 0], [3, 9, 27], [0, 0, 0]]], dtype=np.float64),
        to_dense(features_predictions_labels_list[1].features['sparse_values'][
            encoding.NODE_SUFFIX]))
    self.assertAllEqual(
        np.array([[[2, 4, 8], [0, 0, 0], [5, 25, 125]]], dtype=np.float64),
        to_dense(features_predictions_labels_list[2].features['sparse_values'][
            encoding.NODE_SUFFIX]))
    self.assertAllEqual(
        np.array([[[5, 25, 125], [7, 49, 343], [11, 121, 1331]]],
                 dtype=np.float64),
        to_dense(features_predictions_labels_list[3].features['sparse_values'][
            encoding.NODE_SUFFIX]))

    eval_saved_model.metrics_reset_update_get_list(
        features_predictions_labels_list)
    metric_values = eval_saved_model.get_metric_values()

    self.assertDictElementsAlmostEqual(metric_values, {
        'mean_squared_error': 0.0,
        'mean_prediction': 3203.5,
    })

  def testPredictListMultipleExamplesPerInputModel(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fake_multi_examples_per_input_estimator
        .fake_multi_examples_per_input_estimator(None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    fpls = eval_saved_model.predict_list([b'0', b'1', b'3', b'0', b'2'])
    self.assertEqual(6, len(fpls))

    input_index = []
    example_count = []
    labels = []
    predictions = []
    intra_input_index = []
    annotation = []

    def _check_and_append_feature(feature_name, one_fpl, feature_values):
      self.assertEqual(
          (1,), one_fpl.features[feature_name][encoding.NODE_SUFFIX].shape)
      feature_values.append(
          one_fpl.features[feature_name][encoding.NODE_SUFFIX][0])

    for fpl in fpls:
      _check_and_append_feature('input_index', fpl, input_index)
      _check_and_append_feature('example_count', fpl, example_count)
      _check_and_append_feature('intra_input_index', fpl, intra_input_index)
      _check_and_append_feature('annotation', fpl, annotation)

      self.assertAllEqual((1,), fpl.labels[encoding.DEFAULT_LABELS_DICT_KEY][
          encoding.NODE_SUFFIX].shape)
      labels.append(
          fpl.labels[encoding.DEFAULT_LABELS_DICT_KEY][encoding.NODE_SUFFIX][0])

      self.assertAllEqual(
          (1,), fpl.predictions['predictions'][encoding.NODE_SUFFIX].shape)
      predictions.append(
          fpl.predictions['predictions'][encoding.NODE_SUFFIX][0])

    self.assertSequenceEqual([1, 3, 3, 3, 2, 2], example_count)
    self.assertSequenceEqual([1, 2, 2, 2, 4, 4], input_index)
    self.assertSequenceEqual([0, 0, 1, 2, 0, 1], intra_input_index)
    self.assertAllEqual([
        b'raw_input: 1; index: 0', b'raw_input: 3; index: 0',
        b'raw_input: 3; index: 1', b'raw_input: 3; index: 2',
        b'raw_input: 2; index: 0', b'raw_input: 2; index: 1'
    ], annotation)

    self.assertSequenceEqual([1, 2, 2, 2, 4, 4], labels)
    self.assertSequenceEqual([1, 2, 2, 2, 4, 4], predictions)

  def testPredictListMultipleExamplesPerInputModelNoExampleInInput(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fake_multi_examples_per_input_estimator
        .fake_multi_examples_per_input_estimator(None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    fpls = eval_saved_model.predict_list(['0', '0'])
    self.assertFalse(fpls)

  def testPredictListMisalignedExampleRef(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fake_multi_examples_per_input_estimator
        .bad_multi_examples_per_input_estimator_misaligned_example_ref(
            None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    with self.assertRaisesRegexp(ValueError,
                                 'example_ref should be batch-aligned'):
      eval_saved_model.predict_list(['1'])

  def testPredictListOutOfRangeExampleRef(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fake_multi_examples_per_input_estimator
        .bad_multi_examples_per_input_estimator_out_of_range_example_ref(
            None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    with self.assertRaisesRegexp(ValueError,
                                 'An index in example_ref is out of range'):
      eval_saved_model.predict_list(['1'])

  def testVariablePredictionLengths(self):
    # Check that we can handle cases where the model produces predictions of
    # different lengths for different examples.
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_classifier.simple_fixed_prediction_classifier(
            None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    _, prediction_dict, _ = (
        eval_saved_model.get_features_predictions_labels_dicts())
    with eval_saved_model.graph_as_default():
      eval_saved_model.register_additional_metric_ops({
          'total_non_trivial_classes':
              metrics.total(
                  tf.reduce_sum(
                      tf.cast(
                          tf.logical_and(
                              tf.not_equal(prediction_dict['classes'], '?'),
                              tf.not_equal(prediction_dict['classes'], '')),
                          tf.int32))),
          'example_count':
              metrics.total(tf.shape(prediction_dict['classes'])[0]),
          'total_score':
              metrics.total(prediction_dict['probabilities']),
      })

    example1 = self._makeExample(classes=['apple'], scores=[100.0])
    example2 = self._makeExample()
    example3 = self._makeExample(
        classes=['durian', 'elderberry', 'fig', 'grape'],
        scores=[300.0, 301.0, 302.0, 303.0])
    example4 = self._makeExample(
        classes=['banana', 'cherry'], scores=[400.0, 401.0])

    fpl_list1 = self.predict_injective_example_list(eval_saved_model, [
        example1.SerializeToString(),
        example2.SerializeToString(),
    ])
    fpl_list2 = self.predict_injective_example_list(eval_saved_model, [
        example3.SerializeToString(),
        example4.SerializeToString(),
    ])

    # Note that the '?' and 0 default values come from the model.
    self.assertAllEqual(
        np.array([[b'apple']]),
        fpl_list1[0].predictions['classes'][encoding.NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[100]]),
        fpl_list1[0].predictions['probabilities'][encoding.NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[b'?']]),
        fpl_list1[1].predictions['classes'][encoding.NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[0]]),
        fpl_list1[1].predictions['probabilities'][encoding.NODE_SUFFIX])

    self.assertAllEqual(
        np.array([[b'durian', b'elderberry', b'fig', b'grape']]),
        fpl_list2[0].predictions['classes'][encoding.NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[300, 301, 302, 303]]),
        fpl_list2[0].predictions['probabilities'][encoding.NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[b'banana', b'cherry', b'?', b'?']]),
        fpl_list2[1].predictions['classes'][encoding.NODE_SUFFIX])
    self.assertAllEqual(
        np.array([[400, 401, 0, 0]]),
        fpl_list2[1].predictions['probabilities'][encoding.NODE_SUFFIX])

    eval_saved_model.metrics_reset_update_get_list(fpl_list1 + fpl_list2)
    metric_values = eval_saved_model.get_metric_values()

    self.assertDictElementsAlmostEqual(
        metric_values, {
            'total_non_trivial_classes': 7.0,
            'example_count': 4.0,
            'total_score': 2107.0,
        })

  def testEvaluateExistingMetricsCSVInputBasic(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        csv_linear_classifier.simple_csv_linear_classifier(
            None, temp_eval_export_dir))

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, '3.0,english,1.0')
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, '3.0,chinese,0.0')
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(metric_values, {
        'accuracy': 1.0,
        'auc': 1.0
    })

  def testEvaluateExistingMetricsCustomEstimatorBasic(self):
    # Custom estimator aims to predict age * 3 + 1
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = custom_estimator.simple_custom_estimator(
        None, temp_eval_export_dir)

    example1 = example_pb2.Example()
    example1.features.feature['age'].float_list.value[:] = [1.0]
    example1.features.feature['label'].float_list.value[:] = [3.0]
    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    example2 = example_pb2.Example()
    example2.features.feature['age'].float_list.value[:] = [2.0]
    example2.features.feature['label'].float_list.value[:] = [7.0]
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()

    # We don't control the trained model's weights fully, but it should
    # predict close to what it aims to. The "target" mean prediction is 5.5.
    self.assertIn('mean_prediction', metric_values)
    self.assertGreater(metric_values['mean_prediction'], 5.4)
    self.assertLess(metric_values['mean_prediction'], 5.6)

    # The "target" mean absolute error is 0.5
    self.assertIn('mean_absolute_error', metric_values)
    self.assertGreater(metric_values['mean_absolute_error'], 0.4)
    self.assertLess(metric_values['mean_absolute_error'], 0.6)

    self.assertHasKeyWithValueAlmostEqual(metric_values, 'mean_label', 5.0)

  def testEvaluateWithAdditionalMetricsBasic(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = multi_head.simple_multi_head(None,
                                                      temp_eval_export_dir)

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    _, prediction_dict, label_dict = (
        eval_saved_model.get_features_predictions_labels_dicts())
    with eval_saved_model.graph_as_default():
      metric_ops = {}
      value_op, update_op = tf.metrics.mean_absolute_error(
          label_dict['english_head'][0][0],
          prediction_dict['english_head/probabilities'][0][1])
      metric_ops['mean_absolute_error/english_head'] = (value_op, update_op)

      value_op, update_op = tf.contrib.metrics.count(
          prediction_dict['english_head/logits'])
      metric_ops['example_count/english_head'] = (value_op, update_op)

      eval_saved_model.register_additional_metric_ops(metric_ops)

    example1 = self._makeMultiHeadExample('english')
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    example2 = self._makeMultiHeadExample('chinese')
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()

    # Check that the original metrics are still there.
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'accuracy/english_head': 1.0,
            'accuracy/chinese_head': 1.0,
            'accuracy/other_head': 1.0,
            'auc/english_head': 1.0,
            'auc/chinese_head': 1.0,
            'auc/other_head': 1.0,
            'label/mean/english_head': 0.5,
            'label/mean/chinese_head': 0.5,
            'label/mean/other_head': 0.0
        })

    # Check the added metrics.
    # We don't control the trained model's weights fully, but it should
    # predict probabilities > 0.7.
    self.assertIn('mean_absolute_error/english_head', metric_values)
    self.assertLess(metric_values['mean_absolute_error/english_head'], 0.3)

    self.assertHasKeyWithValueAlmostEqual(metric_values,
                                          'example_count/english_head', 2.0)

  def testGetAndSetMetricVariables(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = multi_head.simple_multi_head(None,
                                                      temp_eval_export_dir)

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    _, prediction_dict, _ = (
        eval_saved_model.get_features_predictions_labels_dicts())
    with eval_saved_model.graph_as_default():
      metric_ops = {}
      value_op, update_op = tf.contrib.metrics.count(
          prediction_dict['english_head/logits'])
      metric_ops['example_count/english_head'] = (value_op, update_op)

      eval_saved_model.register_additional_metric_ops(metric_ops)

    example1 = self._makeMultiHeadExample('english')
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)
    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'label/mean/english_head': 1.0,
            'label/mean/chinese_head': 0.0,
            'label/mean/other_head': 0.0,
            'example_count/english_head': 1.0
        })
    metric_variables = eval_saved_model.get_metric_variables()

    example2 = self._makeMultiHeadExample('chinese')
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)
    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'label/mean/english_head': 0.5,
            'label/mean/chinese_head': 0.5,
            'label/mean/other_head': 0.0,
            'example_count/english_head': 2.0
        })

    # Now set metric variables to what they were after the first example.
    eval_saved_model.set_metric_variables(metric_variables)
    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'label/mean/english_head': 1.0,
            'label/mean/chinese_head': 0.0,
            'label/mean/other_head': 0.0,
            'example_count/english_head': 1.0
        })

  def testResetMetricVariables(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = multi_head.simple_multi_head(None,
                                                      temp_eval_export_dir)

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    _, prediction_dict, _ = (
        eval_saved_model.get_features_predictions_labels_dicts())
    with eval_saved_model.graph_as_default():
      metric_ops = {}
      value_op, update_op = tf.contrib.metrics.count(
          prediction_dict['english_head/logits'])
      metric_ops['example_count/english_head'] = (value_op, update_op)

      eval_saved_model.register_additional_metric_ops(metric_ops)

    example1 = self._makeMultiHeadExample('english')
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)
    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'label/mean/english_head': 1.0,
            'label/mean/chinese_head': 0.0,
            'label/mean/other_head': 0.0,
            'example_count/english_head': 1.0
        })
    eval_saved_model.reset_metric_variables()

    example2 = self._makeMultiHeadExample('chinese')
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)
    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'label/mean/english_head': 0.0,
            'label/mean/chinese_head': 1.0,
            'label/mean/other_head': 0.0,
            'example_count/english_head': 1.0
        })

  def testMetricsResetUpdateGetList(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = multi_head.simple_multi_head(None,
                                                      temp_eval_export_dir)

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    _, prediction_dict, _ = (
        eval_saved_model.get_features_predictions_labels_dicts())
    with eval_saved_model.graph_as_default():
      metric_ops = {}
      value_op, update_op = tf.contrib.metrics.count(
          prediction_dict['english_head/logits'])
      metric_ops['example_count/english_head'] = (value_op, update_op)

      eval_saved_model.register_additional_metric_ops(metric_ops)

    example1 = self._makeMultiHeadExample('english')
    features_predictions_labels1 = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    metric_variables1 = eval_saved_model.metrics_reset_update_get(
        features_predictions_labels1)

    example2 = self._makeMultiHeadExample('chinese')
    features_predictions_labels2 = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    metric_variables2 = eval_saved_model.metrics_reset_update_get(
        features_predictions_labels2)

    example3 = self._makeMultiHeadExample('other')
    features_predictions_labels3 = self.predict_injective_single_example(
        eval_saved_model, example3.SerializeToString())
    metric_variables3 = eval_saved_model.metrics_reset_update_get(
        features_predictions_labels3)

    eval_saved_model.set_metric_variables(metric_variables1)
    metric_values1 = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values1, {
            'label/mean/english_head': 1.0,
            'label/mean/chinese_head': 0.0,
            'label/mean/other_head': 0.0,
            'example_count/english_head': 1.0
        })

    eval_saved_model.set_metric_variables(metric_variables2)
    metric_values2 = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values2, {
            'label/mean/english_head': 0.0,
            'label/mean/chinese_head': 1.0,
            'label/mean/other_head': 0.0,
            'example_count/english_head': 1.0
        })

    eval_saved_model.set_metric_variables(metric_variables3)
    metric_values3 = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values3, {
            'label/mean/english_head': 0.0,
            'label/mean/chinese_head': 0.0,
            'label/mean/other_head': 1.0,
            'example_count/english_head': 1.0
        })

    eval_saved_model.metrics_reset_update_get_list([
        features_predictions_labels1, features_predictions_labels2,
        features_predictions_labels3
    ])
    metric_values_combined = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values_combined, {
            'label/mean/english_head': 1.0 / 3.0,
            'label/mean/chinese_head': 1.0 / 3.0,
            'label/mean/other_head': 1.0 / 3.0,
            'example_count/english_head': 3.0
        })

  def testEvaluateExistingMetricsWithExportedCustomMetrics(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = linear_classifier.simple_linear_classifier(
        None, temp_eval_export_dir)

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(age=3.0, language='english', label=1.0)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    example2 = self._makeExample(age=2.0, language='chinese', label=0.0)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values, {
            'accuracy': 1.0,
            'auc': 1.0,
            'my_mean_age': 2.5,
            'my_mean_label': 0.5,
            'my_mean_age_times_label': 1.5
        })

    self.assertIn('my_mean_prediction', metric_values)
    self.assertIn('prediction/mean', metric_values)
    self.assertAlmostEqual(
        metric_values['prediction/mean'],
        metric_values['my_mean_prediction'],
        places=5)

  def testServingGraphAlsoExportedIfSpecified(self):
    # Most of the example trainers also pass serving_input_receiver_fn to
    # export_eval_savedmodel, so the serving graph should be included.
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = (
        fixed_prediction_estimator.simple_fixed_prediction_estimator(
            None, temp_eval_export_dir))

    # Check the eval graph.
    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(prediction=0.9, label=0.0)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(metric_values, {'average_loss': 0.81})

    # Check the serving graph.
    estimator = tf.contrib.estimator.SavedModelEstimator(eval_export_dir)

    def predict_input_fn():
      return {'inputs': tf.constant([example1.SerializeToString()])}

    predictions = next(estimator.predict(predict_input_fn))
    self.assertAllClose(predictions['outputs'], np.array([0.9]))

  def testEvaluateExistingMetricsWithExportedCustomMetricsDNN(self):
    temp_eval_export_dir = self._getEvalExportDir()
    _, eval_export_dir = dnn_classifier.simple_dnn_classifier(
        None, temp_eval_export_dir)

    eval_saved_model = load.EvalSavedModel(eval_export_dir)
    example1 = self._makeExample(age=3.0, language='english', label=1.0)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example1.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    example2 = self._makeExample(age=2.0, language='chinese', label=0.0)
    features_predictions_labels = self.predict_injective_single_example(
        eval_saved_model, example2.SerializeToString())
    eval_saved_model.perform_metrics_update(features_predictions_labels)

    metric_values = eval_saved_model.get_metric_values()
    self.assertDictElementsAlmostEqual(
        metric_values,
        {
            # We don't check accuracy and AUC here because it varies from run to
            # run due to DNN initialization
            'my_mean_age': 2.5,
            'my_mean_label': 0.5,
            'my_mean_age_times_label': 1.5
        })

    self.assertIn('my_mean_prediction', metric_values)
    self.assertIn('prediction/mean', metric_values)
    self.assertAlmostEqual(
        metric_values['prediction/mean'],
        metric_values['my_mean_prediction'],
        places=5)


if __name__ == '__main__':
  tf.test.main()
