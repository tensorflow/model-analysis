# Copyright 2022 Google LLC
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

import dataclasses

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_model_analysis.experimental import dataframe
from tensorflow_model_analysis.proto import metrics_for_slice_pb2

from google.protobuf import text_format


class MetricsAsDataFrameTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.metrics_for_slices = [
        text_format.Parse(
            """
        slice_key {
           single_slice_keys {
             column: "age"
             float_value: 38.0
           }
           single_slice_keys {
             column: "sex"
             bytes_value: "Female"
           }
         }
         metric_keys_and_values {
           key {
             name: "mean_absolute_error"
             example_weighted {
             }
           }
           value {
             double_value {
               value: 0.1
             }
           }
         }
         metric_keys_and_values {
           key {
             name: "mean_squared_logarithmic_error"
             example_weighted {
             }
           }
           value {
             double_value {
             value: 0.02
             }
           }
         }
         """, metrics_for_slice_pb2.MetricsForSlice()),
        text_format.Parse(
            """
         slice_key {}
         metric_keys_and_values {
           key {
             name: "mean_absolute_error"
           }
           value {
             double_value {
               value: 0.3
             }
           }
         }
         """, metrics_for_slice_pb2.MetricsForSlice())
    ]

    self.metrics_overall_slice_only = [
        text_format.Parse(
            """
         slice_key {}
         metric_keys_and_values {
           key {
             name: "mean_absolute_error"
           }
           value {
             double_value {
               value: 0.3
             }
           }
         }
         metric_keys_and_values {
           key {
             name: "example_count"
           }
           value {
             double_value {
               value: 10
             }
           }
         }
         """, metrics_for_slice_pb2.MetricsForSlice())
    ]

  def testLoadMetricsAsDataFrame_DoubleValueOnly(self):
    dfs = dataframe.metrics_as_dataframes(self.metrics_for_slices)

    expected = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0, None],
        ('slices', 'sex'): [b'Female', b'Female', None],
        ('slices', 'Overall'): [None, None, ''],
        ('metric_keys', 'name'): [
            'mean_absolute_error', 'mean_squared_logarithmic_error',
            'mean_absolute_error'
        ],
        ('metric_keys', 'model_name'): ['', '', ''],
        ('metric_keys', 'output_name'): ['', '', ''],
        ('metric_keys', 'example_weighted'): [False, False, None],
        ('metric_keys', 'is_diff'): [False, False, False],
        ('metric_values', 'double_value'): [0.1, 0.02, 0.3],
    })
    pd.testing.assert_frame_equal(expected, dfs.double_value)

  def testLoadMetricsAsDataFrame_DoubleValueIncludeEmptyColumn(self):
    dfs = dataframe.metrics_as_dataframes(
        self.metrics_for_slices, include_empty_columns=True)
    expected = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0, None],
        ('slices', 'sex'): [b'Female', b'Female', None],
        ('slices', 'Overall'): [None, None, ''],
        ('metric_keys', 'name'): [
            'mean_absolute_error', 'mean_squared_logarithmic_error',
            'mean_absolute_error'
        ],
        ('metric_keys', 'model_name'): ['', '', ''],
        ('metric_keys', 'output_name'): ['', '', ''],
        ('metric_keys', 'sub_key'): [None, None, None],
        ('metric_keys', 'aggregation_type'): [None, None, None],
        ('metric_keys', 'example_weighted'): [False, False, None],
        ('metric_keys', 'is_diff'): [False, False, False],
        ('metric_values', 'double_value'): [0.1, 0.02, 0.3],
    })
    pd.testing.assert_frame_equal(expected, dfs.double_value)

  def testLoadMetricsAsDataFrame_DoubleValueOverallSliceOnly(self):
    dfs = dataframe.metrics_as_dataframes(
        self.metrics_overall_slice_only, include_empty_columns=False)
    expected = pd.DataFrame({
        ('slices', 'Overall'): ['', ''],
        ('metric_keys', 'name'): ['mean_absolute_error', 'example_count'],
        ('metric_keys', 'model_name'): ['', ''],
        ('metric_keys', 'output_name'): ['', ''],
        ('metric_keys', 'is_diff'): [False, False],
        ('metric_values', 'double_value'): [0.3, 10],
    })
    pd.testing.assert_frame_equal(expected, dfs.double_value)

  def testLoadMetricsAsDataFrame_Empty(self):
    metrics_for_slices = [
        text_format.Parse(
            """
        slice_key {
           single_slice_keys {
             column: "age"
             float_value: 38.0
           }
           single_slice_keys {
             column: "sex"
             bytes_value: "Female"
           }
         }
         """, metrics_for_slice_pb2.MetricsForSlice()),
    ]
    dfs = dataframe.metrics_as_dataframes(metrics_for_slices)
    self.assertTrue(all(d is None for d in dataclasses.astuple(dfs)))

  def testAutoPivot_MetricsDataFrame(self):
    df = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0, None],
        ('slices', 'sex'): [b'Female', b'Female', None],
        ('metric_keys', 'name'): [
            'mean_absolute_error', 'mean_squared_logarithmic_error',
            'mean_absolute_error'
        ],
        ('metric_keys', 'model_name'): ['', '', ''],
        ('metric_keys', 'output_name'): ['', '', ''],
        ('metric_keys', 'example_weighted'): [False, False, None],
        ('metric_keys', 'is_diff'): [False, False, False],
        ('metric_values', 'double_value'): [0.1, 0.02, 0.3],
    })
    df = dataframe.auto_pivot(
        df, stringify_slices=False, collapse_column_names=False)
    mux = pd.MultiIndex.from_tuples(
        [
            (('metric_values', 'double_value'), False, 'mean_absolute_error'),
            (
                ('metric_values', 'double_value'),
                False,
                'mean_squared_logarithmic_error',
            ),
            (('metric_values', 'double_value'), np.nan, 'mean_absolute_error'),
        ],
        names=(
            None,
            ('metric_keys', 'example_weighted'),
            ('metric_keys', 'name'),
        ),
    )
    mix = pd.MultiIndex.from_tuples(
        [(np.nan, np.nan), (38.0, b'Female')],
        names=[('slices', 'age'), ('slices', 'sex')],
    )
    expected = pd.DataFrame(
        [[np.nan, np.nan, 0.3], [0.1, 0.02, np.nan]],
        index=mix,
        columns=mux,
    )
    pd.testing.assert_frame_equal(expected, df, check_column_type=False)

  def testAutoPivot_MetricsDataFrameStringifySlices(self):
    df = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0, None],
        ('slices', 'sex'): [b'Female', b'Female', None],
        ('metric_keys', 'name'): [
            'mean_absolute_error', 'mean_squared_logarithmic_error',
            'mean_absolute_error'
        ],
        ('metric_keys', 'model_name'): ['', '', ''],
        ('metric_keys', 'output_name'): ['', '', ''],
        ('metric_keys', 'example_weighted'): [False, False, None],
        ('metric_keys', 'is_diff'): [False, False, False],
        ('metric_values', 'double_value'): [0.1, 0.02, 0.3],
    })
    df = dataframe.auto_pivot(
        df, stringify_slices=True, collapse_column_names=False)
    mux = pd.MultiIndex.from_tuples(
        [
            (('metric_values', 'double_value'), np.nan, 'mean_absolute_error'),
            (('metric_values', 'double_value'), False, 'mean_absolute_error'),
            (
                ('metric_values', 'double_value'),
                False,
                'mean_squared_logarithmic_error',
            ),
        ],
        names=(
            None,
            ('metric_keys', 'example_weighted'),
            ('metric_keys', 'name'),
        ),
    )
    index = pd.Index(
        ['', "age:38.0; sex:b'Female'"], dtype='object', name='slices'
    )
    expected = pd.DataFrame(
        [[0.3, np.nan, np.nan], [np.nan, 0.1, 0.02]],
        index=index,
        columns=mux,
    )
    pd.testing.assert_frame_equal(expected, df, check_column_type=False)

  def testAutoPivot_MetricsDataFrameCollapseColumnNames(self):
    df = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0, None],
        ('slices', 'sex'): [b'Female', b'Female', None],
        ('metric_keys', 'name'): [
            'mean_absolute_error',
            'mean_squared_logarithmic_error',
            'mean_absolute_error',
        ],
        ('metric_keys', 'model_name'): ['', '', ''],
        ('metric_keys', 'output_name'): ['', '', ''],
        ('metric_keys', 'example_weighted'): [False, False, None],
        ('metric_keys', 'is_diff'): [False, False, False],
        ('metric_values', 'double_value'): [0.1, 0.02, 0.3],
    })
    df = dataframe.auto_pivot(
        df, stringify_slices=False, collapse_column_names=True)
    mux = pd.MultiIndex.from_tuples(
        [
            (False, 'mean_absolute_error'),
            (False, 'mean_squared_logarithmic_error'),
            (np.nan, 'mean_absolute_error'),
        ],
        names=[('metric_keys', 'example_weighted'), ('metric_keys', 'name')],
    )
    mix = pd.MultiIndex.from_tuples(
        [(np.nan, np.nan), (38.0, b'Female')],
        names=[('slices', 'age'), ('slices', 'sex')],
    )
    expected = pd.DataFrame(
        [[np.nan, np.nan, 0.3], [0.1, 0.02, np.nan]],
        index=mix,
        columns=mux,
    )
    pd.testing.assert_frame_equal(expected, df, check_column_type=False)

  def testAutoPivot_MetricsDataFrameOverallSliceOnly(self):
    dfs = dataframe.metrics_as_dataframes(
        self.metrics_overall_slice_only, include_empty_columns=False)
    df = dfs.double_value
    expected = df.pivot(
        index=[
            ('slices', 'Overall'),
        ],
        columns=[('metric_keys', 'name')],
        values=[('metric_values', 'double_value')])
    df = dataframe.auto_pivot(
        df, stringify_slices=False, collapse_column_names=False)
    pd.testing.assert_frame_equal(expected, df, check_column_type=False)


class PlotsAsDataFrameTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.plots_for_slice = [
        text_format.Parse(
            """
        slice_key {
           single_slice_keys {
             column: "age"
             float_value: 38.0
           }
           single_slice_keys {
             column: "sex"
             bytes_value: "Female"
           }
         }
         plot_keys_and_values {
           key {
             example_weighted {
             }
           }
           value {
             confusion_matrix_at_thresholds {
               matrices {
                 threshold: 0.5
                 false_negatives: 10
                 true_negatives: 10
                 false_positives: 10
                 true_positives: 10
                 precision: 0.9
                 recall: 0.8
               }
               matrices {
                 threshold: 0.5
                 false_negatives: 10
                 true_negatives: 10
                 false_positives: 10
                 true_positives: 10
                 precision: 0.9
                 recall: 0.8
               }
             }
           }
         }
         """, metrics_for_slice_pb2.PlotsForSlice())
    ]

  def testLoadPlotsAsDataFrame(self):
    dfs = dataframe.plots_as_dataframes(self.plots_for_slice)
    expected = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0],
        ('slices', 'sex'): [b'Female', b'Female'],
        ('plot_keys', 'name'): ['', ''],
        ('plot_keys', 'model_name'): ['', ''],
        ('plot_keys', 'output_name'): ['', ''],
        ('plot_keys', 'example_weighted'): [False, False],
        ('plot_data', 'threshold'): [0.5, 0.5],
        ('plot_data', 'false_negatives'): [10.0, 10.0],
        ('plot_data', 'true_negatives'): [10.0, 10.0],
        ('plot_data', 'false_positives'): [10.0, 10.0],
        ('plot_data', 'true_positives'): [10.0, 10.0],
        ('plot_data', 'precision'): [0.9, 0.9],
        ('plot_data', 'recall'): [0.8, 0.8],
        ('plot_data', 'false_positive_rate'): [0.0, 0.0],
        ('plot_data', 'f1'): [0.0, 0.0],
        ('plot_data', 'accuracy'): [0.0, 0.0],
        ('plot_data', 'false_omission_rate'): [0.0, 0.0],
    })
    pd.testing.assert_frame_equal(expected, dfs.confusion_matrix_at_thresholds)

  def testLoadPlotsAsDataFrame_IncludeEmptyColumn(self):
    dfs = dataframe.plots_as_dataframes(
        self.plots_for_slice, include_empty_columns=True)
    expected = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0],
        ('slices', 'sex'): [b'Female', b'Female'],
        ('plot_keys', 'name'): ['', ''],
        ('plot_keys', 'model_name'): ['', ''],
        ('plot_keys', 'output_name'): ['', ''],
        ('plot_keys', 'sub_key'): [None, None],
        ('plot_keys', 'example_weighted'): [False, False],
        ('plot_data', 'threshold'): [0.5, 0.5],
        ('plot_data', 'false_negatives'): [10.0, 10.0],
        ('plot_data', 'true_negatives'): [10.0, 10.0],
        ('plot_data', 'false_positives'): [10.0, 10.0],
        ('plot_data', 'true_positives'): [10.0, 10.0],
        ('plot_data', 'precision'): [0.9, 0.9],
        ('plot_data', 'recall'): [0.8, 0.8],
        ('plot_data', 'false_positive_rate'): [0.0, 0.0],
        ('plot_data', 'f1'): [0.0, 0.0],
        ('plot_data', 'accuracy'): [0.0, 0.0],
        ('plot_data', 'false_omission_rate'): [0.0, 0.0],
    })
    pd.testing.assert_frame_equal(expected, dfs.confusion_matrix_at_thresholds)

  def testLoadPlotsAsDataFrame_Empty(self):
    plots_for_slice = [
        text_format.Parse(
            """
        slice_key {
           single_slice_keys {
             column: "age"
             float_value: 38.0
           }
           single_slice_keys {
             column: "sex"
             bytes_value: "Female"
           }
         }
         """, metrics_for_slice_pb2.PlotsForSlice())
    ]

    dfs = dataframe.plots_as_dataframes(plots_for_slice)
    self.assertIsNone(dfs.confusion_matrix_at_thresholds)

  def testAutoPivot_PlotsDataFrame(self):
    dfs = dataframe.plots_as_dataframes(self.plots_for_slice)
    df = dataframe.auto_pivot(
        dfs.confusion_matrix_at_thresholds, stringify_slices=False)
    expected = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0],
        ('slices', 'sex'): [b'Female', b'Female'],
        ('plot_keys', 'name'): ['', ''],
        ('plot_keys', 'model_name'): ['', ''],
        ('plot_keys', 'output_name'): ['', ''],
        ('plot_keys', 'example_weighted'): [False, False],
        ('plot_data', 'threshold'): [0.5, 0.5],
        ('plot_data', 'false_negatives'): [10.0, 10.0],
        ('plot_data', 'true_negatives'): [10.0, 10.0],
        ('plot_data', 'false_positives'): [10.0, 10.0],
        ('plot_data', 'true_positives'): [10.0, 10.0],
        ('plot_data', 'precision'): [0.9, 0.9],
        ('plot_data', 'recall'): [0.8, 0.8],
        ('plot_data', 'false_positive_rate'): [0.0, 0.0],
        ('plot_data', 'f1'): [0.0, 0.0],
        ('plot_data', 'accuracy'): [0.0, 0.0],
        ('plot_data', 'false_omission_rate'): [0.0, 0.0],
    }).pivot(
        index=[('slices', 'age'), ('slices', 'sex')],
        columns=[],
        values=[
            ('plot_data', 'threshold'),
            ('plot_data', 'false_negatives'),
            ('plot_data', 'true_negatives'),
            ('plot_data', 'false_positives'),
            ('plot_data', 'true_positives'),
            ('plot_data', 'precision'),
            ('plot_data', 'recall'),
            ('plot_data', 'false_positive_rate'),
            ('plot_data', 'f1'),
            ('plot_data', 'accuracy'),
            ('plot_data', 'false_omission_rate'),
        ],
    )
    pd.testing.assert_frame_equal(expected, df, check_column_type=False)

  def testAutoPivot_PlotsDataFrameCollapseColumnNames(self):
    dfs = dataframe.plots_as_dataframes(self.plots_for_slice)
    df = dataframe.auto_pivot(
        dfs.confusion_matrix_at_thresholds,
        stringify_slices=False,
        collapse_column_names=True)
    expected = pd.DataFrame({
        ('slices', 'age'): [38.0, 38.0],
        ('slices', 'sex'): [b'Female', b'Female'],
        ('plot_keys', 'name'): ['', ''],
        ('plot_keys', 'model_name'): ['', ''],
        ('plot_keys', 'output_name'): ['', ''],
        ('plot_keys', 'example_weighted'): [False, False],
        ('plot_data', 'threshold'): [0.5, 0.5],
        ('plot_data', 'false_negatives'): [10.0, 10.0],
        ('plot_data', 'true_negatives'): [10.0, 10.0],
        ('plot_data', 'false_positives'): [10.0, 10.0],
        ('plot_data', 'true_positives'): [10.0, 10.0],
        ('plot_data', 'precision'): [0.9, 0.9],
        ('plot_data', 'recall'): [0.8, 0.8],
        ('plot_data', 'false_positive_rate'): [0.0, 0.0],
        ('plot_data', 'f1'): [0.0, 0.0],
        ('plot_data', 'accuracy'): [0.0, 0.0],
        ('plot_data', 'false_omission_rate'): [0.0, 0.0],
    }).pivot(
        index=[('slices', 'age'), ('slices', 'sex')],
        columns=[],
        values=[
            ('plot_data', 'threshold'),
            ('plot_data', 'false_negatives'),
            ('plot_data', 'true_negatives'),
            ('plot_data', 'false_positives'),
            ('plot_data', 'true_positives'),
            ('plot_data', 'precision'),
            ('plot_data', 'recall'),
            ('plot_data', 'false_positive_rate'),
            ('plot_data', 'f1'),
            ('plot_data', 'accuracy'),
            ('plot_data', 'false_omission_rate'),
        ],
    )
    pd.testing.assert_frame_equal(expected, df, check_column_type=False)

