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
"""Validator types."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Dict, NamedTuple, Text

import apache_beam as beam

# A validator takes a set of alternative evaluations as input and compares them
# to produce a Validation output. A typical example of a validator is the
# MetricsValidator that compares 'baseline' and 'candidate' Evaluations produced
# by separate runs of the MetricsAndPlotsEvaluator.
Validator = NamedTuple(  # pylint: disable=invalid-name
    'Validator',
    [
        ('stage_name', Text),
        # Dict[Text, Evaluation] -> Validation (e.g. 'baseline', 'candidate').
        ('ptransform', beam.PTransform)
    ])

# A Validation represents the output from verifying alternative Evaluations.
# The validation outputs are keyed by their associated output type. For example,
# the serialized proto from evaluating metrics would be stored under "metrics".
Validation = Dict[Text, beam.pvalue.PCollection]
