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
"""Utility and schema methods for the chicago_taxi sample."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_transform import coders as tft_coders
from tensorflow_transform.tf_metadata import dataset_schema

# Categorical features are assumed to each have a maximum value in the dataset.
MAX_CATEGORICAL_FEATURE_VALUES = [24, 31, 12]
CATEGORICAL_FEATURE_KEYS = [
    'trip_start_hour',
    'trip_start_day',
    'trip_start_month'
]

DENSE_FLOAT_FEATURE_KEYS = [
    'trip_miles',
    'fare',
    'trip_seconds'
]

# Number of buckets used by tf.transform for encoding each feature.
FEATURE_BUCKET_COUNT = 10

BUCKET_FEATURE_KEYS = [
    'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
    'dropoff_longitude'
]

# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
VOCAB_SIZE = 1000

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
OOV_SIZE = 10

VOCAB_FEATURE_KEYS = [
    'pickup_census_tract',
    'dropoff_census_tract',
    'payment_type',
    'company',
    'pickup_community_area',
    'dropoff_community_area'
]
LABEL_KEY = 'tips'
FARE_KEY = 'fare'


# Tf.Transform considers these features as "raw"
def get_raw_feature_spec():
  return {
      'fare':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'trip_start_timestamp':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'trip_start_hour':
          tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
      'trip_start_day':
          tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
      'trip_start_month':
          tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
      'pickup_latitude':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'pickup_longitude':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'dropoff_latitude':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'dropoff_longitude':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'tips':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'trip_miles':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'pickup_census_tract':
          tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
      'dropoff_census_tract':
          tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
      'pickup_community_area':
          tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
      'payment_type':
          tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
      'company':
          tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
      'trip_seconds':
          tf.FixedLenFeature(shape=[], dtype=tf.float32, default_value=0.0),
      'dropoff_community_area':
          tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
  }


def make_csv_coder():
  """Return a coder for tf.transform to read csv files."""
  column_names = [
      'pickup_community_area', 'fare', 'trip_start_month', 'trip_start_hour',
      'trip_start_day', 'trip_start_timestamp', 'pickup_latitude',
      'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'trip_miles',
      'pickup_census_tract', 'dropoff_census_tract', 'payment_type', 'company',
      'dropoff_community_area', 'tips', 'trip_seconds'
  ]
  parsing_feature_spec = get_raw_feature_spec()
  parsing_schema = dataset_schema.from_feature_spec(parsing_feature_spec)
  return tft_coders.CsvCoder(column_names, parsing_schema)


def clean_raw_data_dict(input_dict):
  output_dict = {}

  raw_feature_spec = get_raw_feature_spec()
  for key in get_raw_feature_spec():
    if key not in input_dict or not input_dict[key]:
      output_dict[key] = raw_feature_spec[key].default_value
    else:
      output_dict[key] = input_dict[key]
  return output_dict


def make_sql(table_name, max_rows=None, for_eval=False):
  """Creates the sql command for pulling data from BigQuery.

  Args:
    table_name: BigQuery table name
    max_rows: if set, limits the number of rows pulled from BigQuery
    for_eval: True if this is for evaluation, false otherwise

  Returns:
    sql command as string
  """
  if for_eval:
    # 1/3 of the dataset used for eval
    where_clause = 'WHERE MOD(FARM_FINGERPRINT(unique_key), 3) = 0'
  else:
    # 2/3 of the dataset used for training
    where_clause = 'WHERE MOD(FARM_FINGERPRINT(unique_key), 3) > 0'

  limit_clause = ''
  if max_rows:
    limit_clause = 'LIMIT {max_rows}'.format(max_rows=max_rows)
  return """
  SELECT
      CAST(pickup_community_area AS string) AS pickup_community_area,
      CAST(dropoff_community_area AS string) AS dropoff_community_area,
      CAST(pickup_census_tract AS string) AS pickup_census_tract,
      CAST(dropoff_census_tract AS string) AS dropoff_census_tract,
      fare,
      EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
      EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
      EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
      UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
      pickup_latitude,
      pickup_longitude,
      dropoff_latitude,
      dropoff_longitude,
      trip_miles,
      payment_type,
      company,
      trip_seconds,
      tips
  FROM `{table_name}`
  {where_clause}
  {limit_clause}
""".format(table_name=table_name,
           where_clause=where_clause,
           limit_clause=limit_clause)
