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
"""Test for predict extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Standard Imports

import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import config
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import dnn_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.eval_saved_model.example_trainers import multi_head
from tensorflow_model_analysis.extractors import predict_extractor_v2

if tf.version.VERSION.split('.')[0] == '1':
  tf.compat.v1.enable_v2_behavior()


class PredictExtractorTest(testutil.TensorflowModelAnalysisTest):

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def testPredictExtractorWithRegressionModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(temp_export_dir, None))

    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(location=export_dir)])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    predict_extractor = predict_extractor_v2.PredictExtractor(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])

    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1'),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string2'),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=1.0,
            fixed_string='fixed_string3')
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | predict_extractor.stage_name >> predict_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got_preds):
        try:
          self.assertLen(got_preds, 3)
          expected_preds = [0.2, 0.8, 0.5]
          for got_pred, expected_pred in zip(got_preds, expected_preds):
            self.assertIn(constants.PREDICTIONS_KEY, got_pred)
            self.assertAlmostEqual(got_pred[constants.PREDICTIONS_KEY],
                                   expected_pred)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictExtractorWithBinaryClassificationModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = dnn_classifier.simple_dnn_classifier(
        temp_export_dir, None, n_classes=2)

    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(location=export_dir)])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    predict_extractor = predict_extractor_v2.PredictExtractor(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])

    examples = [
        self._makeExample(age=1.0, language='english', label=0),
        self._makeExample(age=2.0, language='chinese', label=1),
        self._makeExample(age=3.0, language='chinese', label=0),
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | predict_extractor.stage_name >> predict_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 3)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for pred_key in ('logistic', 'probabilities', 'all_classes'):
              self.assertIn(pred_key, item[constants.PREDICTIONS_KEY])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictExtractorWithMultiClassModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = dnn_classifier.simple_dnn_classifier(
        temp_export_dir, None, n_classes=3)

    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(location=export_dir)])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    predict_extractor = predict_extractor_v2.PredictExtractor(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])

    examples = [
        self._makeExample(age=1.0, language='english', label=0),
        self._makeExample(age=2.0, language='chinese', label=1),
        self._makeExample(age=3.0, language='english', label=2),
        self._makeExample(age=4.0, language='chinese', label=1),
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | predict_extractor.stage_name >> predict_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 4)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for pred_key in ('probabilities', 'all_classes'):
              self.assertIn(pred_key, item[constants.PREDICTIONS_KEY])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictExtractorWithMultiOutputModel(self):
    temp_export_dir = self._getExportDir()
    export_dir, _ = multi_head.simple_multi_head(temp_export_dir, None)

    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(location=export_dir)])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    predict_extractor = predict_extractor_v2.PredictExtractor(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])

    examples = [
        self._makeExample(
            age=1.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=1.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
        self._makeExample(
            age=2.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=2.0,
            language='other',
            english_label=0.0,
            chinese_label=1.0,
            other_label=1.0)
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | predict_extractor.stage_name >> predict_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 4)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for output_name in ('chinese_head', 'english_head', 'other_head'):
              for pred_key in ('logistic', 'probabilities', 'all_classes'):
                self.assertIn(output_name + '/' + pred_key,
                              item[constants.PREDICTIONS_KEY])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictExtractorWithMultiModels(self):
    temp_export_dir = self._getExportDir()
    export_dir1, _ = multi_head.simple_multi_head(temp_export_dir, None)
    export_dir2, _ = multi_head.simple_multi_head(temp_export_dir, None)

    eval_config = config.EvalConfig(model_specs=[
        config.ModelSpec(location=export_dir1, name='model1'),
        config.ModelSpec(location=export_dir2, name='model2')
    ])
    eval_shared_model1 = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir1, tags=[tf.saved_model.SERVING])
    eval_shared_model2 = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir2, tags=[tf.saved_model.SERVING])
    predict_extractor = predict_extractor_v2.PredictExtractor(
        eval_config=eval_config,
        eval_shared_models=[eval_shared_model1, eval_shared_model2])

    examples = [
        self._makeExample(
            age=1.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=1.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0),
        self._makeExample(
            age=2.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0),
        self._makeExample(
            age=2.0,
            language='other',
            english_label=0.0,
            chinese_label=1.0,
            other_label=1.0)
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create([e.SerializeToString() for e in examples])
          | 'InputsToExtracts' >> model_eval_lib.InputsToExtracts()
          | predict_extractor.stage_name >> predict_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 4)
          for item in got:
            # We can't verify the actual predictions, but we can verify the keys
            self.assertIn(constants.PREDICTIONS_KEY, item)
            for model_name in ('model1', 'model2'):
              self.assertIn(model_name, item[constants.PREDICTIONS_KEY])
              for output_name in ('chinese_head', 'english_head', 'other_head'):
                for pred_key in ('logistic', 'probabilities', 'all_classes'):
                  self.assertIn(output_name + '/' + pred_key,
                                item[constants.PREDICTIONS_KEY][model_name])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')

  def testPredictExtractorWithKerasModel(self):
    input1 = tf.keras.layers.Input(shape=(1,), name='input1')
    input2 = tf.keras.layers.Input(shape=(1,), name='input2')
    inputs = [input1, input2]
    input_layer = tf.keras.layers.concatenate(inputs)
    output_layer = tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid, name='output')(
            input_layer)
    model = tf.keras.models.Model(inputs, output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=.001),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    train_features = {'input1': [[0.0], [1.0]], 'input2': [[1.0], [0.0]]}
    labels = [[1], [0]]
    example_weights = [1.0, 0.5]
    dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, labels, example_weights))
    dataset = dataset.shuffle(buffer_size=1).repeat().batch(2)
    model.fit(dataset, steps_per_epoch=1)

    export_dir = self._getExportDir()
    model.save(export_dir, save_format='tf')

    eval_config = config.EvalConfig(
        model_specs=[config.ModelSpec(location=export_dir)])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING])
    predict_extractor = predict_extractor_v2.PredictExtractor(
        eval_config=eval_config, eval_shared_models=[eval_shared_model])

    predict_features = [
        {
            'input1': np.array([0.0], dtype=np.float32),
            'input2': np.array([1.0], dtype=np.float32),
            'non_model_feature': np.array([0]),  # should be ignored by model
        },
        {
            'input1': np.array([1.0], dtype=np.float32),
            'input2': np.array([0.0], dtype=np.float32),
            'non_model_feature': np.array([1]),  # should be ignored by model
        }
    ]

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create' >> beam.Create(predict_features)
          | 'FeaturesToExtracts' >>
          beam.Map(lambda x: {constants.FEATURES_KEY: x})
          | predict_extractor.stage_name >> predict_extractor.ptransform)

      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 2)
          # We can't verify the actual predictions, but we can verify the keys.
          for item in got:
            self.assertIn(constants.PREDICTIONS_KEY, item)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result, label='result')


if __name__ == '__main__':
  tf.test.main()
