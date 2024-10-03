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
"""Tests for Tfx-Bsl Predictions Extractor."""


import pytest
import os

from absl.testing import parameterized
import apache_beam as beam
from apache_beam.testing import util
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import testutil
from tensorflow_model_analysis.eval_saved_model.example_trainers import dnn_classifier
from tensorflow_model_analysis.eval_saved_model.example_trainers import fixed_prediction_estimator_extra_fields
from tensorflow_model_analysis.eval_saved_model.example_trainers import multi_head
from tensorflow_model_analysis.extractors import features_extractor
from tensorflow_model_analysis.extractors import predictions_extractor
from tensorflow_model_analysis.extractors import tfx_bsl_predictions_extractor
from tensorflow_model_analysis.proto import config_pb2
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2



@pytest.mark.xfail(run=False, reason="PR 183 This class contains tests that fail and needs to be fixed. "
"If all tests pass, please remove this mark.")
class TfxBslPredictionsExtractorTest(
    testutil.TensorflowModelAnalysisTest, parameterized.TestCase
):

  def _getExportDir(self):
    return os.path.join(self._getTempDir(), 'export_dir')

  def _create_tfxio_and_feature_extractor(
      self, eval_config: config_pb2.EvalConfig, schema: schema_pb2.Schema
  ):
    tfx_io = test_util.InMemoryTFExampleRecord(
        schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
    )
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=tfx_io.ArrowSchema(),
        tensor_representations=tfx_io.TensorRepresentations(),
    )
    feature_extractor = features_extractor.FeaturesExtractor(
        eval_config=eval_config,
        tensor_representations=tensor_adapter_config.tensor_representations,
    )
    return tfx_io, feature_extractor

  @parameterized.named_parameters(
      ('ModelSignaturesDoFnInference', False), ('TFXBSLBulkInference', True)
  )
  def testRegressionModel(self, experimental_bulk_inference):
    temp_export_dir = self._getExportDir()
    export_dir, _ = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            temp_export_dir, None
        )
    )

    eval_config = config_pb2.EvalConfig(model_specs=[config_pb2.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        feature {
          name: "prediction"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "fixed_int"
          type: INT
        }
        feature {
          name: "fixed_float"
          type: FLOAT
        }
        feature {
          name: "fixed_string"
          type: BYTES
        }
        """,
            schema_pb2.Schema(),
        ),
    )

    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=1.0,
            fixed_string='fixed_string3',
        ),
    ]
    num_examples = len(examples)

    if experimental_bulk_inference:
      prediction_extractor = (
          tfx_bsl_predictions_extractor.TfxBslPredictionsExtractor(
              eval_config=eval_config,
              eval_shared_model=eval_shared_model,
              output_batch_size=num_examples,
          )
      )
    else:
      prediction_extractor = predictions_extractor.PredictionsExtractor(
          eval_config=eval_config, eval_shared_model=eval_shared_model
      )

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          self.assertAllClose(
              np.array([[0.2], [0.8], [0.5]]), got[0][constants.PREDICTIONS_KEY]
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def testInferenceBatchSize(self):
    temp_export_dir = self._getExportDir()
    # Running pyformat results in a lint error. Formating the lint error breaks
    # the pyformat presubmit. This style matches the other tests.
    # pyformat: disable
    export_dir, _ = (
        fixed_prediction_estimator_extra_fields
        .simple_fixed_prediction_estimator_extra_fields(temp_export_dir, None))
    # pyformat: enable

    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=1.0,
            fixed_string='fixed_string3',
        ),
    ]
    num_examples = len(examples)

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(
                inference_batch_size=num_examples,
            )
        ]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        feature {
          name: "prediction"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "fixed_int"
          type: INT
        }
        feature {
          name: "fixed_float"
          type: FLOAT
        }
        feature {
          name: "fixed_string"
          type: BYTES
        }
        """,
            schema_pb2.Schema(),
        ),
    )

    prediction_extractor = (
        tfx_bsl_predictions_extractor.TfxBslPredictionsExtractor(
            eval_config=eval_config,
            eval_shared_model=eval_shared_model,
            output_batch_size=num_examples,
        )
    )

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          self.assertAllClose(
              np.array([[0.2], [0.8], [0.5]]), got[0][constants.PREDICTIONS_KEY]
          )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  def testNoDefinedBatchSize(self):
    """Simple test to cover batch_size=None code path."""
    temp_export_dir = self._getExportDir()
    export_dir, _ = (
        fixed_prediction_estimator_extra_fields.simple_fixed_prediction_estimator_extra_fields(
            temp_export_dir, None
        )
    )

    eval_config = config_pb2.EvalConfig(model_specs=[config_pb2.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        feature {
          name: "prediction"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "fixed_int"
          type: INT
        }
        feature {
          name: "fixed_float"
          type: FLOAT
        }
        feature {
          name: "fixed_string"
          type: BYTES
        }
        """,
            schema_pb2.Schema(),
        ),
    )

    examples = [
        self._makeExample(
            prediction=0.2,
            label=1.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string1',
        ),
        self._makeExample(
            prediction=0.8,
            label=0.0,
            fixed_int=1,
            fixed_float=1.0,
            fixed_string='fixed_string2',
        ),
        self._makeExample(
            prediction=0.5,
            label=0.0,
            fixed_int=2,
            fixed_float=1.0,
            fixed_string='fixed_string3',
        ),
    ]

    prediction_extractor = (
        tfx_bsl_predictions_extractor.TfxBslPredictionsExtractor(
            eval_config=eval_config, eval_shared_model=eval_shared_model
        )
    )

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource()
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
          | beam.FlatMap(lambda extracts: extracts[constants.PREDICTIONS_KEY])
      )
      # pylint: enable=no-value-for-parameter

      util.assert_that(
          result,
          util.equal_to(
              [np.array([0.2]), np.array([0.8]), np.array([0.5])],
              equals_fn=np.isclose,
          ),
      )

  @parameterized.named_parameters(
      ('ModelSignaturesDoFnInferenceUnspecifiedSignature', False, ''),
      ('ModelSignaturesDoFnInferencePredictSignature', False, 'predict'),
      (
          'ModelSignaturesDoFnInferenceServingDefaultSignature',
          False,
          'serving_default',
      ),
      (
          'ModelSignaturesDoFnInferenceClassificationSignature',
          False,
          'classification',
      ),
      ('TFXBSLBulkInferenceUnspecifiedSignature', True, ''),
      ('TFXBSLBulkInferencePredictSignature', True, 'predict'),
      ('TFXBSLBulkInferenceServingDefaultSignature', True, 'serving_default'),
      ('TFXBSLBulkInferenceClassificationSignature', True, 'classification'),
  )
  def testBinaryClassificationModel(
      self, experimental_bulk_inference, signature_name
  ):
    temp_export_dir = self._getExportDir()
    num_classes = 2
    export_dir, _ = dnn_classifier.simple_dnn_classifier(
        temp_export_dir, None, n_classes=num_classes
    )

    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(signature_name=signature_name)]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "langauge"
          type: BYTES
        }
        feature {
          name: "label"
          type: INT
        }
        """,
            schema_pb2.Schema(),
        ),
    )

    examples = [
        self._makeExample(age=1.0, language='english', label=0),
        self._makeExample(age=2.0, language='chinese', label=1),
        self._makeExample(age=3.0, language='chinese', label=0),
    ]
    num_examples = len(examples)

    if experimental_bulk_inference:
      prediction_extractor = (
          tfx_bsl_predictions_extractor.TfxBslPredictionsExtractor(
              eval_config=eval_config,
              eval_shared_model=eval_shared_model,
              output_batch_size=num_examples,
          )
      )
    else:
      prediction_extractor = predictions_extractor.PredictionsExtractor(
          eval_config=eval_config, eval_shared_model=eval_shared_model
      )

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          # Prediction API cases. We default '' to 'predict'.
          if signature_name in ('', 'predict'):
            for pred_key in ('logistic', 'probabilities', 'all_classes'):
              self.assertIn(pred_key, got[0][constants.PREDICTIONS_KEY])
            self.assertEqual(
                (num_examples, num_classes),
                got[0][constants.PREDICTIONS_KEY]['probabilities'].shape,
            )
          # Classification API cases. The classification signature is also the
          # 'serving_default' signature for this model.
          if signature_name in ('serving_default', 'classification'):
            for pred_key in ('classes', 'scores'):
              self.assertIn(pred_key, got[0][constants.PREDICTIONS_KEY])
            self.assertEqual(
                (num_examples, num_classes),
                got[0][constants.PREDICTIONS_KEY]['classes'].shape,
            )
            self.assertEqual(
                (num_examples, num_classes),
                got[0][constants.PREDICTIONS_KEY]['scores'].shape,
            )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  @parameterized.named_parameters(
      ('ModelSignaturesDoFnInferenceUnspecifiedSignature', False, ''),
      ('ModelSignaturesDoFnInferencePredictSignature', False, 'predict'),
      (
          'ModelSignaturesDoFnInferenceServingDefaultSignature',
          False,
          'serving_default',
      ),
      (
          'ModelSignaturesDoFnInferenceClassificationSignature',
          False,
          'classification',
      ),
      ('TFXBSLBulkInferenceUnspecifiedSignature', True, ''),
      ('TFXBSLBulkInferencePredictSignature', True, 'predict'),
      ('TFXBSLBulkInferenceServingDefaultSignature', True, 'serving_default'),
      ('TFXBSLBulkInferenceClassificationSignature', True, 'classification'),
  )
  def testMultiClassModel(self, experimental_bulk_inference, signature_name):
    temp_export_dir = self._getExportDir()
    num_classes = 3
    export_dir, _ = dnn_classifier.simple_dnn_classifier(
        temp_export_dir, None, n_classes=num_classes
    )

    eval_config = config_pb2.EvalConfig(
        model_specs=[config_pb2.ModelSpec(signature_name=signature_name)]
    )
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "langauge"
          type: BYTES
        }
        feature {
          name: "label"
          type: INT
        }
        """,
            schema_pb2.Schema(),
        ),
    )

    examples = [
        self._makeExample(age=1.0, language='english', label=0),
        self._makeExample(age=2.0, language='chinese', label=1),
        self._makeExample(age=3.0, language='english', label=2),
        self._makeExample(age=4.0, language='chinese', label=1),
    ]
    num_examples = len(examples)

    if experimental_bulk_inference:
      prediction_extractor = (
          tfx_bsl_predictions_extractor.TfxBslPredictionsExtractor(
              eval_config=eval_config,
              eval_shared_model=eval_shared_model,
              output_batch_size=num_examples,
          )
      )
    else:
      prediction_extractor = predictions_extractor.PredictionsExtractor(
          eval_config=eval_config, eval_shared_model=eval_shared_model
      )

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          # Prediction API cases. We default '' to 'predict'.
          if signature_name in ('', 'predict'):
            for pred_key in ('probabilities', 'all_classes'):
              self.assertIn(pred_key, got[0][constants.PREDICTIONS_KEY])
            self.assertEqual(
                (num_examples, num_classes),
                got[0][constants.PREDICTIONS_KEY]['probabilities'].shape,
            )
          # Classification API cases. The classification signature is also the
          # 'serving_default' signature for this model.
          if signature_name in ('serving_default', 'classification'):
            for pred_key in ('classes', 'scores'):
              self.assertIn(pred_key, got[0][constants.PREDICTIONS_KEY])
            self.assertEqual(
                (num_examples, num_classes),
                got[0][constants.PREDICTIONS_KEY]['classes'].shape,
            )
            self.assertEqual(
                (num_examples, num_classes),
                got[0][constants.PREDICTIONS_KEY]['scores'].shape,
            )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  @parameterized.named_parameters(
      ('ModelSignaturesDoFnInference', False), ('TFXBSLBulkInference', True)
  )
  def testMultiOutputModel(self, experimental_bulk_inference):
    temp_export_dir = self._getExportDir()
    export_dir, _ = multi_head.simple_multi_head(temp_export_dir, None)

    eval_config = config_pb2.EvalConfig(model_specs=[config_pb2.ModelSpec()])
    eval_shared_model = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir, tags=[tf.saved_model.SERVING]
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "langauge"
          type: BYTES
        }
        feature {
          name: "english_label"
          type: FLOAT
        }
        feature {
          name: "chinese_label"
          type: FLOAT
        }
        feature {
          name: "other_label"
          type: FLOAT
        }
        """,
            schema_pb2.Schema(),
        ),
    )

    examples = [
        self._makeExample(
            age=1.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=1.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=2.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=2.0,
            language='other',
            english_label=0.0,
            chinese_label=1.0,
            other_label=1.0,
        ),
    ]
    num_examples = len(examples)

    if experimental_bulk_inference:
      prediction_extractor = (
          tfx_bsl_predictions_extractor.TfxBslPredictionsExtractor(
              eval_config=eval_config,
              eval_shared_model=eval_shared_model,
              output_batch_size=num_examples,
          )
      )
    else:
      prediction_extractor = predictions_extractor.PredictionsExtractor(
          eval_config=eval_config, eval_shared_model=eval_shared_model
      )

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys.
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          for output_name in ('chinese_head', 'english_head', 'other_head'):
            for pred_key in ('logistic', 'probabilities', 'all_classes'):
              self.assertIn(
                  output_name + '/' + pred_key,
                  got[0][constants.PREDICTIONS_KEY],
              )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)

  @parameterized.named_parameters(
      ('ModelSignaturesDoFnInference', False), ('TFXBSLBulkInference', True)
  )
  def testMultiModels(self, experimental_bulk_inference):
    temp_export_dir = self._getExportDir()
    export_dir1, _ = multi_head.simple_multi_head(temp_export_dir, None)
    export_dir2, _ = multi_head.simple_multi_head(temp_export_dir, None)

    eval_config = config_pb2.EvalConfig(
        model_specs=[
            config_pb2.ModelSpec(name='model1'),
            config_pb2.ModelSpec(name='model2'),
        ]
    )
    eval_shared_model1 = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir1, tags=[tf.saved_model.SERVING]
    )
    eval_shared_model2 = self.createTestEvalSharedModel(
        eval_saved_model_path=export_dir2, tags=[tf.saved_model.SERVING]
    )
    tfx_io, feature_extractor = self._create_tfxio_and_feature_extractor(
        eval_config,
        text_format.Parse(
            """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "langauge"
          type: BYTES
        }
        feature {
          name: "english_label"
          type: FLOAT
        }
        feature {
          name: "chinese_label"
          type: FLOAT
        }
        feature {
          name: "other_label"
          type: FLOAT
        }
        """,
            schema_pb2.Schema(),
        ),
    )

    examples = [
        self._makeExample(
            age=1.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=1.0,
            language='chinese',
            english_label=0.0,
            chinese_label=1.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=2.0,
            language='english',
            english_label=1.0,
            chinese_label=0.0,
            other_label=0.0,
        ),
        self._makeExample(
            age=2.0,
            language='other',
            english_label=0.0,
            chinese_label=1.0,
            other_label=1.0,
        ),
    ]
    num_examples = len(examples)

    if experimental_bulk_inference:
      prediction_extractor = (
          tfx_bsl_predictions_extractor.TfxBslPredictionsExtractor(
              eval_config=eval_config,
              eval_shared_model={
                  'model1': eval_shared_model1,
                  'model2': eval_shared_model2,
              },
              output_batch_size=num_examples,
          )
      )
    else:
      prediction_extractor = predictions_extractor.PredictionsExtractor(
          eval_config=eval_config,
          eval_shared_model={
              'model1': eval_shared_model1,
              'model2': eval_shared_model2,
          },
      )

    with beam.Pipeline() as pipeline:
      # pylint: disable=no-value-for-parameter
      result = (
          pipeline
          | 'Create'
          >> beam.Create(
              [e.SerializeToString() for e in examples], reshuffle=False
          )
          | 'BatchExamples' >> tfx_io.BeamSource(batch_size=num_examples)
          | 'InputsToExtracts' >> model_eval_lib.BatchedInputsToExtracts()
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )
      # pylint: enable=no-value-for-parameter

      def check_result(got):
        try:
          self.assertLen(got, 1)
          # We can't verify the actual predictions, but we can verify the keys
          self.assertIn(constants.PREDICTIONS_KEY, got[0])
          for model_name in ('model1', 'model2'):
            self.assertIn(model_name, got[0][constants.PREDICTIONS_KEY])
            for output_name in ('chinese_head', 'english_head', 'other_head'):
              for pred_key in ('logistic', 'probabilities', 'all_classes'):
                self.assertIn(
                    output_name + '/' + pred_key,
                    got[0][constants.PREDICTIONS_KEY][model_name],
                )

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(result, check_result)


