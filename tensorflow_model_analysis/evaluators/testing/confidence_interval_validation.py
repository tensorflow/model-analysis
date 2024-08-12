# Copyright 2022 Google LLC
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
"""Binary for validating confidence interval behavior on synthetic datasets."""

import collections
import os
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import tensorflow as tf
from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.eval_saved_model import util
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tfx_bsl.tfxio import tf_example_record

from google.protobuf import text_format
# from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.example import example_pb2

_BINARY_CLASSIFICATION_SCENARIO = 'BINARY_CLASSIFICATION'
_REGRESSION_SCENARIO = 'REGRESSION'
_SCENARIOS = [_BINARY_CLASSIFICATION_SCENARIO, _REGRESSION_SCENARIO]

flags.DEFINE_enum(
    'scenario',
    None,
    _SCENARIOS,
    'The scenario to validate, where the '
    'scenario encodes the task type and example generation logic',
)
flags.DEFINE_enum(
    'methodology',
    None,
    ['JACKKNIFE', 'POISSON_BOOTSTRAP'],
    'The CI methodology to use',
)
flags.DEFINE_integer(
    'num_trials',
    None,
    'number of datasets to generate and TFMA runs to perform',
    lower_bound=0,
)
flags.DEFINE_integer(
    'num_examples_per_trial',
    None,
    'number of examples to generate in each trial dataset',
    lower_bound=0,
)
flags.DEFINE_string(
    'output_dir', None, 'existing dir in which to write results'
)
flags.DEFINE_string(
    'pipeline_options',
    '',
    'Command line flags to use in constructing the Beam pipeline options. '
    'For example, "--runner=DirectRunner,--streaming=True"',
)
FLAGS = flags.FLAGS

_ExampleGeneratorType = Callable[[int], Iterable[example_pb2.Example]]
_CIType = Tuple[float, float]
_POPULATION_OUTPUT_NAME = 'population'


def get_regression_scenario() -> (
    Tuple[config_pb2.EvalConfig, _ExampleGeneratorType]
):
  """Returns an EvalConfig and example generator for regression."""
  eval_config = text_format.Parse(
      """
      model_specs {
         label_key: "label"
         prediction_key: "prediction"
      }
      slicing_specs: {}
      metrics_specs {
        metrics { class_name: "MeanSquaredError" }
        metrics { class_name: "Calibration" }
      }
      """,
      config_pb2.EvalConfig(),
  )

  def generate_regression_examples(
      num_examples,
  ) -> Iterator[example_pb2.Example]:
    for _ in range(num_examples):
      yield util.make_example(
          label=float(np.random.random()), prediction=float(np.random.uniform())
      )

  return eval_config, generate_regression_examples


def get_binary_classification_scenario() -> (
    Tuple[config_pb2.EvalConfig, _ExampleGeneratorType]
):
  """Returns an EvalConfig and example generator for binary classification."""
  eval_config = text_format.Parse(
      """
      model_specs {
         label_key: "label"
         prediction_key: "prediction"
      }
      slicing_specs: {}
      metrics_specs {
        metrics { class_name: "AUC" }
        metrics { class_name: "BinaryPrecision" }
      }
      """,
      config_pb2.EvalConfig(),
  )

  def generate_classification_examples(
      num_examples,
  ) -> Iterator[example_pb2.Example]:
    for _ in range(num_examples):
      yield util.make_example(
          label=float(np.random.choice([0, 1])),
          prediction=float(np.random.uniform()),
      )

  return eval_config, generate_classification_examples


def compute_cis(
    scenario: str,
    methodology: str,
    num_trials: int,
    num_examples_per_trial: int,
    output_dir: str,
) -> None:
  """Computes a collection of CIs and the population values for a scenario."""
  if scenario == _BINARY_CLASSIFICATION_SCENARIO:
    eval_config, example_gen_fn = get_binary_classification_scenario()
  elif scenario == _REGRESSION_SCENARIO:
    eval_config, example_gen_fn = get_regression_scenario()
  else:
    raise ValueError(
        f'Unexpected scenario {scenario}. Expected one of {_SCENARIOS}'
    )
  eval_config.options.compute_confidence_intervals.value = True
  eval_config.options.confidence_intervals.method = (
      config_pb2.ConfidenceIntervalOptions.ConfidenceIntervalMethod.Value(
          methodology
      )
  )
  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(',')
  )
  with beam.Pipeline(options=pipeline_options) as pipeline:
    tfx_io = tf_example_record.TFExampleBeamRecord(
        physical_format='generated',
        raw_record_column_name=constants.ARROW_INPUT_COLUMN,
    )
    inputs_per_trial = []
    for i in range(num_trials):
      inputs = (
          pipeline
          | f'CreateExamples[{i}]'
          >> beam.Create(example_gen_fn(num_examples_per_trial))
          | f'Serialize[{i}]'
          >> beam.Map(lambda example: example.SerializeToString())
          | f'BatchExamples[{i}]' >> tfx_io.BeamSource()
      )
      inputs_per_trial.append(inputs)

      trial_output_dir = os.path.join(output_dir, str(i))
      _ = (
          inputs
          | f'Evaluate[{i}]'
          >> model_eval_lib.ExtractEvaluateAndWriteResults(
              eval_config=eval_config, output_path=trial_output_dir
          )
      )
    population_output_dir = os.path.join(output_dir, _POPULATION_OUTPUT_NAME)
    _ = (
        inputs_per_trial
        | 'FlattenInputs' >> beam.Flatten()
        | 'EvaluatePopulation'
        >> model_eval_lib.ExtractEvaluateAndWriteResults(
            eval_config=eval_config, output_path=population_output_dir
        )
    )


def load_point_estimates(
    trial_output_dir: str,
) -> Dict[metric_types.MetricKey, float]:
  """Loads the point estimates for each metric in a TFMA run."""
  population_values = {}
  path = os.path.join(trial_output_dir, 'metrics')
  for rec in tf.compat.v1.io.tf_record_iterator(path):
    metrics_for_slice = metrics_for_slice_pb2.MetricsForSlice.FromString(rec)
    for kv in metrics_for_slice.metric_keys_and_values:
      if kv.value.WhichOneof('type') != 'double_value':
        continue
      population_values[metric_types.MetricKey.from_proto(kv.key)] = (
          kv.value.double_value.value
      )
  return population_values


def load_trial_cis(
    trial_output_dir: str,
) -> Dict[metric_types.MetricKey, _CIType]:
  """Loads the CI (lower, upper) for each metric in a TFMA run."""
  trial_cis = {}
  path = os.path.join(trial_output_dir, 'metrics')
  for rec in tf.compat.v1.io.tf_record_iterator(path):
    metrics_for_slice = metrics_for_slice_pb2.MetricsForSlice.FromString(rec)
    for kv in metrics_for_slice.metric_keys_and_values:
      if kv.value.WhichOneof('type') != 'double_value':
        continue
      lower = kv.confidence_interval.lower_bound.double_value.value
      upper = kv.confidence_interval.upper_bound.double_value.value
      trial_cis[metric_types.MetricKey.from_proto(kv.key)] = (lower, upper)
  return trial_cis


def load_cis(
    output_dir: str,
) -> Tuple[
    Dict[metric_types.MetricKey, List[_CIType]],
    Dict[metric_types.MetricKey, float],
]:
  """Loads the population point estimates and trial CIs from TFMA runs."""
  population_values = load_point_estimates(
      os.path.join(output_dir, _POPULATION_OUTPUT_NAME)
  )
  trials_cis = collections.defaultdict(list)
  pattern = os.path.join(output_dir, 'trial-*')
  for trial_path in tf.io.gfile.glob(pattern):
    trial_cis = load_trial_cis(trial_path)
    for key, ci in trial_cis.items():
      trials_cis[key].append(ci)
  return trials_cis, population_values


def compute_coverage(output_dir: str) -> Dict[metric_types.MetricKey, float]:
  """Computes the per-metric CI coverage fraction."""
  trial_cis, population_values = load_cis(output_dir)
  coverage_counts = collections.defaultdict(int)
  for metric_name, cis in trial_cis.items():
    for lower, upper in cis:
      coverage_counts[metric_name] += int(
          population_values[metric_name] >= lower
          and population_values[metric_name] <= upper
      )

  coverage_rates = {
      k: count / len(trial_cis[k]) for k, count in coverage_counts.items()
  }
  return coverage_rates


def main(argv: Sequence[str]) -> None:
  del argv
  compute_cis(
      scenario=FLAGS.scenario,
      methodology=FLAGS.methodology,
      num_trials=FLAGS.num_trials,
      num_examples_per_trial=FLAGS.num_examples_per_trial,
      output_dir=FLAGS.output_dir,
  )
  coverage_rates = compute_coverage(FLAGS.output_dir)
  print(coverage_rates)


if __name__ == '__main__':
  app.run(main)
