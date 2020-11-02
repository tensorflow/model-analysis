# Lint as: python3
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
"""Init module for TensorFlow Model Analysis slicer."""

from tensorflow_model_analysis.slicer.slicer_lib import deserialize_slice_key
from tensorflow_model_analysis.slicer.slicer_lib import FanoutSlices
from tensorflow_model_analysis.slicer.slicer_lib import OVERALL_SLICE_KEY
from tensorflow_model_analysis.slicer.slicer_lib import serialize_slice_key
from tensorflow_model_analysis.slicer.slicer_lib import SingleSliceSpec
from tensorflow_model_analysis.slicer.slicer_lib import stringify_slice_key
