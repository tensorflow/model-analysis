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
"""Init module for TensorFlow Model Analysis extractors."""

from tensorflow_model_analysis.extractors import legacy_meta_feature_extractor as meta_feature_extractor
from tensorflow_model_analysis.extractors.example_weights_extractor import ExampleWeightsExtractor
from tensorflow_model_analysis.extractors.extractor import Extractor
from tensorflow_model_analysis.extractors.extractor import Filter
from tensorflow_model_analysis.extractors.features_extractor import FeaturesExtractor
from tensorflow_model_analysis.extractors.labels_extractor import LabelsExtractor
from tensorflow_model_analysis.extractors.legacy_feature_extractor import FeatureExtractor
from tensorflow_model_analysis.extractors.legacy_input_extractor import InputExtractor
from tensorflow_model_analysis.extractors.legacy_predict_extractor import PredictExtractor
from tensorflow_model_analysis.extractors.predictions_extractor import PredictionsExtractor
from tensorflow_model_analysis.extractors.slice_key_extractor import SLICE_KEY_EXTRACTOR_STAGE_NAME
from tensorflow_model_analysis.extractors.slice_key_extractor import SliceKeyExtractor
