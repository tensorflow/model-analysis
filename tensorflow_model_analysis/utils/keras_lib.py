# Copyright 2023 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Imports keras 2."""
import tensorflow as tf  # pylint:disable=unused-import

# import keras 2
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
  import tf_keras  # pylint: disable=g-import-not-at-top,unused-import
  from tf_keras.api._v1 import keras as tf_keras_v1  # pylint: disable=g-import-not-at-top,unused-import
  from tf_keras.api._v2 import keras as tf_keras_v2  # pylint: disable=g-import-not-at-top,unused-import
else:
  tf_keras = tf.keras  # Keras 2
  tf_keras_v1 = tf.compat.v1.keras
  tf_keras_v2 = tf.compat.v2.keras
