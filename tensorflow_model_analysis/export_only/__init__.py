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
"""Init module for TensorFlow Model Analysis export-only modules.

This module contains only the export-related modules of TFMA, for use in your
Trainer. You may want to use this in your Trainer as it has a smaller import
footprint (it doesn't include Apache Beam which is required for analysis but not
for export). To evaluate your model using TFMA, you should use the full TFMA
module instead, i.e. import tensorflow_model_analysis

Example usage:
  import tensorflow_model_analysis.export_only as tfma_export

  def eval_input_receiver_fn():
    ...
    return tfma_export.export.EvalInputReceiver(...)

  tfma_export.export.export_eval_saved_model(...)
"""

from tensorflow_model_analysis.eval_saved_model import export
from tensorflow_model_analysis.eval_saved_model import exporter
