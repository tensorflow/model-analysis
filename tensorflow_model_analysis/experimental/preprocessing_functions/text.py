# Copyright 2021 Google LLC
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
"""Text related preprocessing functions."""

import re
import string
import tensorflow as tf

_ESCAPED_PUNCTUATIONS = re.escape(string.punctuation)


@tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
def whitespace_tokenization(input_data):
  standardized = tf.strings.regex_replace(
      tf.strings.lower(input_data), '[%s]' % _ESCAPED_PUNCTUATIONS, '')
  tokens = tf.strings.split(standardized)
  return tf.map_fn(fn=lambda t: tf.unique(t)[0], elems=tokens)
