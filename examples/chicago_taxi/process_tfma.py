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
"""Runs a batch job for performing Tensorflow Model Analysis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tempfile

import apache_beam as beam

import tensorflow as tf
import tensorflow_model_analysis as tfma

from tensorflow_model_analysis.eval_saved_model.post_export_metrics import post_export_metrics
from tensorflow_model_analysis.slicer import slicer
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_schema

from trainer import taxi


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--eval_model_dir',
      help='Input path to the model which will be evaluated.')
  parser.add_argument(
      '--eval_result_dir',
      help='Output directory in which the model analysis result is written.')
  parser.add_argument(
      '--big_query_table',
      help='BigQuery path to input examples which will be evaluated.')
  parser.add_argument(
      '--input_csv',
      help='CSV file containing raw data which will be evaluated.')
  parser.add_argument(
      '--max_eval_rows',
      help='Maximum number of rows to evaluate on.',
      default=None,
      type=int)

  known_args, pipeline_args = parser.parse_known_args()

  if known_args.eval_result_dir:
    eval_result_dir = known_args.eval_result_dir
  else:
    eval_result_dir = tempfile.mkdtemp()

  slice_spec = [
      slicer.SingleSliceSpec(),
      slicer.SingleSliceSpec(columns=['trip_start_hour'])
  ]

  with beam.Pipeline(argv=pipeline_args) as pipeline:
    if known_args.input_csv:
      csv_coder = taxi.make_csv_coder()
      raw_data = (
          pipeline
          | 'ReadFromText' >> beam.io.ReadFromText(
              known_args.input_csv, skip_header_lines=1)
          | 'ParseCSV' >> beam.Map(csv_coder.decode))
    elif known_args.big_query_table:
      query = taxi.make_sql(
          known_args.big_query_table, known_args.max_eval_rows, for_eval=True)
      raw_data = (
          pipeline
          | 'ReadBigQuery' >> beam.io.Read(
              beam.io.BigQuerySource(query=query, use_standard_sql=True)))
    else:
      raise ValueError('one of --input_csv or --big_query_table should be '
                       'provided.')

    # Examples must be in clean tf-example format.
    raw_feature_spec = taxi.get_raw_feature_spec()
    raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
    coder = example_proto_coder.ExampleProtoCoder(raw_schema)

    _ = (
        raw_data
        | 'CleanData' >> beam.Map(taxi.clean_raw_data_dict)
        | 'ToSerializedTFExample' >> beam.Map(coder.encode)
        | 'EvaluateAndWriteResults' >> tfma.EvaluateAndWriteResults(
            eval_saved_model_path=known_args.eval_model_dir,
            slice_spec=slice_spec,
            add_metrics_callbacks=[
                post_export_metrics.calibration_plot_and_prediction_histogram(),
                post_export_metrics.auc_plots()
            ],
            output_path=eval_result_dir,
            desired_batch_size=100))


if __name__ == '__main__':
  main()
