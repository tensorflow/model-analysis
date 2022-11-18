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
"""Contains functions to operate bounding boxes.

It includes functions for bounding boxes filtering, area calculations, sorting,
etc.
"""

import collections
from typing import List, Optional

import numpy as np
from tensorflow_model_analysis.utils import util


def stack_labels(extracts: util.StandardExtracts,
                 col_names: List[str],
                 model_name: Optional[str] = None,
                 allow_missing_key: Optional[bool] = False) -> np.ndarray:
  """Stacks several numpy arrays in the extracts into a single one for labels.

  It will search for column_names in labels, features and transformed features.
  If not found, it will raise an error.

  Examples:
    Extracts
    {
        features: {
            'xmin': [0, 0, 0.1]
            'xmax': [1, 1, 0.5]
            'ymin_max': [[0.2, 1], [0.3, 1], [0.1, 1]]
        }
    }
    stack_labels(extracts, ['xmin', 'xmax']) ==
      np.array([[0, 1], [0, 1], [0.1, 0.5]])
    stack_labels(extracts, ['xmin', 'xmax', 'ymin_max']) =
      np.array([[0, 1, 0.2, 1], [0, 1, 0.3, 1], [0.1, 0.5, 0.1 ,1]])

  Args:
   extracts: TFMA extracts that stores the keys.
   col_names: Keys of columns which will be stacked.
   model_name: The name of the model for outputs.
   allow_missing_key: (Optional) If true, it will return empty array instead of
     raising errors when col_names are not found.

  Returns:
   A numpy array that stacks all the columns together.

  Raises:
   KeyError: The columns for stacking are not found in extracts.
   ValueError: The format of the input is not valid for stacking.
  """
  cols = []

  dict_to_search = collections.ChainMap(
      extracts.get_labels(model_name) or {},
      extracts.get_features() or {},
      extracts.get_transformed_features(model_name) or {})

  for col_name in col_names:
    if dict_to_search and col_name in dict_to_search:
      new_cols = dict_to_search[col_name]
      if new_cols.ndim == 2:
        cols.append(new_cols)
      elif new_cols.ndim == 1:
        cols.append(new_cols[:, np.newaxis])
      else:
        raise ValueError(f"Dimension of input under {col_name}"
                         " should be 1 or 2.")
    else:
      if allow_missing_key:
        return np.empty((0, 5))
      else:
        raise KeyError(f"Key {col_name} is not found under labels, "
                       "features, or transformed features of the extracts."
                       "Please set allow_missing_key to True, if you want to "
                       "return empty array instead.")
  result = np.hstack(cols)
  return result


def stack_predictions(extracts: util.StandardExtracts,
                      col_names: List[str],
                      model_name: Optional[str] = None,
                      allow_missing_key: Optional[bool] = False) -> np.ndarray:
  """Stacks several numpy arrays in the extracts into a single predictions.

  It will search for column_names in labels, features and transformed features.
  If not found, it will raise an error.

  Examples:
    Extracts
    {
        features: {
            'xmin': [0, 0, 0.1]
            'xmax': [1, 1, 0.5]
            'ymin_max': [[0.2, 1], [0.3, 1], [0.1, 1]]
        }
    }
    stack_predictions(extracts, ['xmin', 'xmax', 'ymin_max']) =
      np.array([[0, 1, 0.2, 1], [0, 1, 0.3, 1], [0.1, 0.5, 0.1 ,1]])

  Args:
   extracts: TFMA extracts that stores the keys.
   col_names: Keys of columns which will be stacked.
   model_name: The name of the model for outputs.
   allow_missing_key: (Optional) If true, it will return empty array instead of
     raising errors when col_names are not found.

  Returns:
   A numpy array that stacks all the columns together.

  Raises:
   KeyError: The columns for stacking are not found in extracts.
   ValueError: The format of the input is not valid for stacking.
  """
  cols = []

  dict_to_search = collections.ChainMap(
      extracts.get_predictions(model_name) or {},
      extracts.get_features() or {},
      extracts.get_transformed_features(model_name) or {})

  for col_name in col_names:
    if dict_to_search and col_name in dict_to_search:
      new_cols = dict_to_search[col_name]
      if new_cols.ndim == 2:
        cols.append(new_cols)
      elif new_cols.ndim == 1:
        cols.append(new_cols[:, np.newaxis])
      else:
        raise ValueError(f"Dimension of input under {col_name} is "
                         f"{new_cols.ndim}, but should be 1 or 2.")
    else:
      if allow_missing_key:
        return np.empty((0, 6))
      else:
        raise KeyError(f"Key {col_name} is not found under predictions, "
                       "features, or transformed features of the extracts."
                       "Please set allow_missing_key to True, if you want to "
                       "return empty array instead.")
  result = np.hstack(cols)
  return result


def truncate_by_num_detections(
    extracts: util.StandardExtracts,
    num_rows_key: str,
    array_to_truncate: np.ndarray,
    model_name: Optional[str] = None,
    allow_missing_key: Optional[bool] = False,
) -> np.ndarray:
  """Get the array to be truncated by the number of rows.

  Args:
   extracts: TFMA extracts that stores the keys.
   num_rows_key: Number of rows in each column except the paddings. For
     multi-dimensional input, it will truncate on the first dimension.
   array_to_truncate: the array to be truncated te
   model_name: The name of the model for outputs.
   allow_missing_key: (Optional) If true, it will do nothing instead of
     raising errors when col_names are not found.

  Returns:
   The array truncated by the number of rows.

  Raises:
   KeyError: The num_rows_key is not found in extracts.
  """
  num_of_rows = None

  dict_to_search = collections.ChainMap(
      extracts.get_predictions(model_name) or {},
      extracts.get_features() or {},
      extracts.get_transformed_features(model_name) or {})

  if num_rows_key:
    if dict_to_search and num_rows_key in dict_to_search:
      num_of_rows = dict_to_search[num_rows_key]
      if isinstance(num_of_rows, np.ndarray):
        num_of_rows = num_of_rows.item()
    else:
      if not allow_missing_key:
        raise KeyError(f"Key {num_rows_key} is not found under predictions, "
                       "features, or transformed features of the extracts."
                       "Please set allow_missing_key to True, if you want to "
                       "skip truncation instead.")
  result = array_to_truncate
  if num_of_rows and num_of_rows > 0 and len(result) > num_of_rows:
    result = result[:num_of_rows]
  return result
