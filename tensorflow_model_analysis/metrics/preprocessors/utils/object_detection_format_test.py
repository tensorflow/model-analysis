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
"""Tests for iou."""
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.metrics.preprocessors.utils import object_detection_format
from tensorflow_model_analysis.utils import util

_STACK_SPLITFORMAT = util.StandardExtracts({
    constants.FEATURES_KEY: {
        'xmin': np.array([30, 50]),
        'ymin': np.array([100, 100]),
        'xmax': np.array([70, 80]),
        'ymax': np.array([300, 200]),
        'class_id': np.array([0, 0]),
        'num_detections': np.array([1])
    },
})

_STACK_GROUPFORMAT = util.StandardExtracts({
    constants.LABELS_KEY: {
        'bbox': np.array([[30, 100, 70, 300], [50, 100, 80, 200]]),
        'class_id': np.array([0, 0])
    },
})

_STACK_RESULT = np.array([[30, 100, 70, 300, 0], [50, 100, 80, 200, 0]])


class ObjectDetectionFormatTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_splitted_columns', _STACK_SPLITFORMAT,
       ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'], _STACK_RESULT),
      ('_partially_stacked', _STACK_GROUPFORMAT, ['bbox', 'class_id'
                                                 ], _STACK_RESULT))
  def test_stack_column(self, extracts, column_names, expected_result):
    result = object_detection_format.stack_labels(extracts, column_names)
    np.testing.assert_allclose(result, expected_result)

  def test_stack_predictions(self):
    result = object_detection_format.stack_predictions(
        _STACK_SPLITFORMAT, ['xmin', 'ymin', 'xmax', 'ymax', 'class_id'])
    result = object_detection_format.truncate_by_num_detections(
        _STACK_SPLITFORMAT, 'num_detections', result)
    np.testing.assert_allclose(result, _STACK_RESULT[:1])


