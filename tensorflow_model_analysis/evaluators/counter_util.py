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
"""Utility for evaluator to add / update beam counters."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam

from tensorflow_model_analysis import constants


def update_beam_counters(metrics_callbacks):
  """To update beam counters for all the metrics computed."""
  for callback in metrics_callbacks:
    if hasattr(callback, 'name'):
      metrics_counter = beam.metrics.Metrics.gauge(
          constants.METRICS_NAMESPACE, 'metric_computed_%s' % callback.name)
      metrics_counter.set(1)
