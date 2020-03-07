# Lint as: python3
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
"""Math utilities."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports

from scipy import stats
from tensorflow_model_analysis import types


def calculate_confidence_interval(
    t_distribution_value: types.ValueWithTDistribution):
  """Caculate confidence intervals based 95% confidence level."""
  alpha = 0.05
  std_err = t_distribution_value.sample_standard_deviation
  t_stat = stats.t.ppf(1 - (alpha / 2.0),
                       t_distribution_value.sample_degrees_of_freedom)
  upper_bound = t_distribution_value.sample_mean + t_stat * std_err
  lower_bound = t_distribution_value.sample_mean - t_stat * std_err
  return t_distribution_value.sample_mean, lower_bound, upper_bound
