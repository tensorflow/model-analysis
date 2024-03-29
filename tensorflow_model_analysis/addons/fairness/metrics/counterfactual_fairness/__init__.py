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
"""Init module for TensorFlow Model Analysis Counterfactual Fairness metrics."""

# The import will trigger the metrics registration.
from tensorflow_model_analysis.addons.fairness.metrics.counterfactual_fairness.flip_count import FlipCount
from tensorflow_model_analysis.addons.fairness.metrics.counterfactual_fairness.flip_rate import FlipRate
