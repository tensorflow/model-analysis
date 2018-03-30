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
"""Preprocessor applying tf.transform to the chicago_taxi data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import apache_beam as beam

import tensorflow as tf
import tensorflow_transform as transform

from tensorflow_transform.beam import impl as beam_impl
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema

from trainer import taxi


def transform_data(input_handle,
                   outfile_prefix,
                   working_dir,
                   max_rows=None,
                   pipeline_args=None):
  """The main tf.transform method which analyzes and transforms data.

  Args:
    input_handle: BigQuery table name to process specified as
      DATASET.TABLE or path to csv file with input data.
    outfile_prefix: Filename prefix for emitted transformed examples
    working_dir: Directory in which transformed examples and transform
      function will be emitted.
    max_rows: Number of rows to query from BigQuery
    pipeline_args: additional DataflowRunner or DirectRunner args passed to the
      beam pipeline.
  """

  def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    outputs = {}
    for key in taxi.DENSE_FLOAT_FEATURE_KEYS:
      # Preserve this feature as a dense float, setting nan's to the mean.
      outputs[key] = transform.scale_to_z_score(inputs[key])

    for key in taxi.VOCAB_FEATURE_KEYS:
      # Build a vocabulary for this feature.
      outputs[key] = transform.string_to_int(
          inputs[key], top_k=taxi.VOCAB_SIZE, num_oov_buckets=taxi.OOV_SIZE)

    for key in taxi.BUCKET_FEATURE_KEYS:
      outputs[key] = transform.bucketize(inputs[key], taxi.FEATURE_BUCKET_COUNT)

    for key in taxi.CATEGORICAL_FEATURE_KEYS:
      outputs[key] = inputs[key]

    # Was this passenger a big tipper?
    def convert_label(label):
      taxi_fare = inputs[taxi.FARE_KEY]
      return tf.where(
          tf.is_nan(taxi_fare),
          tf.cast(tf.zeros_like(taxi_fare), tf.int64),
          # Test if the tip was > 20% of the fare.
          tf.cast(
              tf.greater(label, tf.multiply(taxi_fare, tf.constant(0.2))),
              tf.int64))

    outputs[taxi.LABEL_KEY] = transform.apply_function(convert_label,
                                                       inputs[taxi.LABEL_KEY])

    return outputs

  raw_feature_spec = taxi.get_raw_feature_spec()
  raw_schema = dataset_schema.from_feature_spec(raw_feature_spec)
  raw_data_metadata = dataset_metadata.DatasetMetadata(raw_schema)

  with beam.Pipeline(argv=pipeline_args) as pipeline:
    with beam_impl.Context(temp_dir=working_dir):
      if input_handle.lower().endswith('csv'):
        csv_coder = taxi.make_csv_coder()
        raw_data = (
            pipeline
            | 'ReadFromText' >> beam.io.ReadFromText(
                input_handle, skip_header_lines=1)
            | 'ParseCSV' >> beam.Map(csv_coder.decode))
      else:
        query = taxi.make_sql(input_handle, max_rows, for_eval=False)
        raw_data = (
            pipeline
            | 'ReadBigQuery' >> beam.io.Read(
                beam.io.BigQuerySource(query=query, use_standard_sql=True)))

      raw_data |= 'CleanData' >> beam.Map(taxi.clean_raw_data_dict)

      transform_fn = ((raw_data, raw_data_metadata)
                      | 'Analyze' >> beam_impl.AnalyzeDataset(preprocessing_fn))

      _ = (
          transform_fn
          | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(working_dir))

      # Shuffling the data before materialization will improve Training
      # effectiveness downstream.
      shuffled_data = raw_data | 'RandomizeData' >> beam.transforms.Reshuffle()

      (transformed_data, transformed_metadata) = (
          ((shuffled_data, raw_data_metadata), transform_fn)
          | 'Transform' >> beam_impl.TransformDataset())

      coder = example_proto_coder.ExampleProtoCoder(transformed_metadata.schema)
      _ = (
          transformed_data
          | 'SerializeExamples' >> beam.Map(coder.encode)
          | 'WriteExamples' >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, outfile_prefix),
              compression_type=beam.io.filesystem.CompressionTypes.GZIP))


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      help=('Input BigQuery table to process specified as: '
            'DATASET.TABLE or path to csv file with input data.'))

  # for preprocessing
  parser.add_argument(
      '--output_dir',
      help=('Directory in which transformed examples and function '
            'will be emitted.'))

  parser.add_argument(
      '--outfile_prefix',
      help='Filename prefix for emitted transformed examples')

  parser.add_argument(
      '--max_rows',
      help='Number of rows to query from BigQuery',
      default=None,
      type=int)

  known_args, pipeline_args = parser.parse_known_args()
  transform_data(
      input_handle=known_args.input,
      outfile_prefix=known_args.outfile_prefix,
      working_dir=known_args.output_dir,
      max_rows=known_args.max_rows,
      pipeline_args=pipeline_args)


if __name__ == '__main__':
  main()
