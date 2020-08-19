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
"""Utilities for tfjs_predict_extractor."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import os
import shutil
import subprocess
import sys
import tempfile
import urllib

import tensorflow as tf


def get_tfjs_binary():
  """Download and return the path to the tfjs binary."""
  if sys.platform == 'darwin':
    url = 'http://storage.googleapis.com/tfjs-inference/tfjs-inference-macos'
  else:
    url = 'http://storage.googleapis.com/tfjs-inference/tfjs-inference-linux'

  base_path = tempfile.mkdtemp()
  path = os.path.join(base_path, 'binary')
  with urllib.request.urlopen(url) as response:
    with tf.io.gfile.GFile(path, 'w') as file:
      shutil.copyfileobj(response, file)
  subprocess.check_call(['chmod', '+x', path])
  return path
