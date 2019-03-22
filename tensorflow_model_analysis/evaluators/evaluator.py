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
"""Evaluator types."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from tensorflow_model_analysis.extractors import extractor
from typing import Dict, List, NamedTuple, Text

# An evaluator is a PTransform that takes Extracts as input and produces an
# Evaluation as output. A typical example of an evaluator is the
# MetricsAndPlotsEvaluator that takes the 'features', 'labels', and
# 'predictions' extracts from the PredictExtractor and evaluates them using post
# export metrics to produce serialized metrics and plots.
Evaluator = NamedTuple(  # pylint: disable=invalid-name
    'Evaluator',
    [
        ('stage_name', Text),
        # Extractor.stage_name. If None then evaluation is run before any
        # extractors are run. If LAST_EXTRACTOR_STAGE_NAME then evaluation is
        # run after the last extractor has run.
        ('run_after', Text),
        # PTransform Extracts -> Evaluation
        ('ptransform', beam.PTransform)
    ])

# An Evaluation represents the output from evaluating the Extracts at a
# particular point in the pipeline. The evaluation outputs are keyed by their
# associated output type. For example, the serialized protos from evaluating
# metrics and plots might be stored under "metrics" and "plots" respectively.
Evaluation = Dict[Text, beam.pvalue.PCollection]


def verify_evaluator(evaluator: Evaluator,
                     extractors: List[extractor.Extractor]):
  """Verifies evaluator is matched with an extractor.

  Args:
    evaluator: Evaluator to verify.
    extractors: Extractors to use in verification.

  Raises:
    ValueError: If an Extractor cannot be found for the Evaluator.
  """
  if (evaluator.run_after and
      evaluator.run_after != extractor.LAST_EXTRACTOR_STAGE_NAME and
      not any(evaluator.run_after == x.stage_name for x in extractors)):
    raise ValueError(
        'Extractor matching run_after=%s for Evaluator %s not found' %
        (evaluator.run_after, evaluator.stage_name))
