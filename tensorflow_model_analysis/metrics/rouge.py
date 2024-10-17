# Copyright 2023 Google LLC
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
"""ROUGE Metrics."""

import dataclasses
from typing import Dict, Iterable, Optional

from absl import logging
import apache_beam as beam
import nltk
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2

from rouge_score import rouge_scorer
from rouge_score import scoring
from rouge_score import tokenizers


_LOGGING_MESSAGE_TOKENIZER_PREPARER = (
    "Finding or downloading 'punkt' from nltk."
)


# TODO(b/287700355): Add __slots__ to _Accumulator
@dataclasses.dataclass
class _Accumulator:
  weighted_count: float = 0.0
  total_precision: float = 0.0
  total_recall: float = 0.0
  total_fmeasure: float = 0.0


class RougeCombiner(beam.CombineFn):
  """Computes ROUGE Scores."""

  def __init__(
      self,
      rouge_type: str,
      key: metric_types.MetricKey,
      eval_config: config_pb2.EvalConfig,
      model_name: str,
      output_name: str,
      use_stemmer: bool,
      split_summaries: bool,
      tokenizer: tokenizers.Tokenizer,
  ):
    """Initializes ROUGE Combiner.

    Args:
      rouge_type: ROUGE type to calculate.
      key: MetricKey for extract_output().
      eval_config: Eval config.
      model_name: The model for which to compute these metrics.
      output_name: The output name for which to compute these metrics.
      use_stemmer: Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching. This arg is used in the
        DefaultTokenizer, but other tokenizers might or might not choose to use
        this.
      split_summaries: Whether to add newlines between sentences for rougeLsum.
      tokenizer: Tokenizer object which has a tokenize() method.
    """
    self._use_nltks_recommended_sentence_tokenizer = (
        split_summaries and rouge_type == 'rougeLsum'
    )
    self.rouge_type = rouge_type
    self.key = key
    self.eval_config = eval_config
    self.model_name = model_name
    self.output_name = output_name
    self.scorer = rouge_scorer.RougeScorer(
        rouge_types=[rouge_type],
        use_stemmer=use_stemmer,
        split_summaries=split_summaries,
        tokenizer=tokenizer,
    )

  def setup(self):
    if self._use_nltks_recommended_sentence_tokenizer:
      tokenizer_installed = False

      if not tokenizer_installed:
        logging.info(_LOGGING_MESSAGE_TOKENIZER_PREPARER)
        nltk.download('punkt')
        nltk.download('punkt_tab')

  def create_accumulator(self) -> _Accumulator:
    return _Accumulator()

  def add_input(
      self,
      accumulator: _Accumulator,
      element: metric_types.StandardMetricInputs,
  ) -> _Accumulator:
    labels, predictions, example_weights = next(
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self.eval_config,
            model_name=self.model_name,
            output_name=self.output_name,
            example_weighted=True,
            flatten=False,
            require_single_example_weight=True,
        )
    )

    example_weight = example_weights[0]
    accumulator.weighted_count += example_weight

    rouge_scores = self.scorer.score_multi(labels, predictions[0])[
        self.rouge_type
    ]
    accumulator.total_precision += rouge_scores.precision * example_weight
    accumulator.total_recall += rouge_scores.recall * example_weight
    accumulator.total_fmeasure += rouge_scores.fmeasure * example_weight

    return accumulator

  def merge_accumulators(
      self, accumulators: Iterable[_Accumulator]
  ) -> _Accumulator:
    accumulators = iter(accumulators)
    result = next(accumulators)
    for accumulator in accumulators:
      result.weighted_count += accumulator.weighted_count
      result.total_precision += accumulator.total_precision
      result.total_recall += accumulator.total_recall
      result.total_fmeasure += accumulator.total_fmeasure

    return result

  def extract_output(
      self, accumulator: _Accumulator
  ) -> Dict[metric_types.MetricKey, scoring.Score]:
    if accumulator.weighted_count == 0.0:
      return {
          self.key: scoring.Score(
              precision=float('nan'), recall=float('nan'), fmeasure=float('nan')
          )
      }
    avg_precision = accumulator.total_precision / accumulator.weighted_count
    avg_recall = accumulator.total_recall / accumulator.weighted_count
    avg_fmeasure = accumulator.total_fmeasure / accumulator.weighted_count
    return {
        self.key: scoring.Score(
            precision=avg_precision, recall=avg_recall, fmeasure=avg_fmeasure
        )
    }


def _rouge(
    rouge_type: str,
    name: str,
    eval_config: config_pb2.EvalConfig,
    model_name: str,
    output_name: str,
    use_stemmer: bool,
    split_summaries: bool,
    tokenizer: tokenizers.Tokenizer,
) -> metric_types.MetricComputations:
  """Returns metric computations for ROUGE."""
  key = metric_types.MetricKey(name=name)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=None,
          combiner=RougeCombiner(
              rouge_type=rouge_type,
              key=key,
              eval_config=eval_config,
              model_name=model_name,
              output_name=output_name,
              use_stemmer=use_stemmer,
              split_summaries=split_summaries,
              tokenizer=tokenizer,
          ),
      )
  ]


class Rouge(metric_types.Metric):
  """ROUGE Metrics.

  ROUGE stands for Recall-Oriented Understudy for Gisting Evaluation. It
  includes measures to automatically determine the quality of a summary by
  comparing it to other (ideal) reference / target summaries.

  ROUGE was originally introduced in the paper:

  Lin, Chin-Yew. ROUGE: a Package for Automatic Evaluation of Summaries. In
  Proceedings of the Workshop on Text Summarization Branches Out (WAS 2004),
  Barcelona, Spain, July 25 - 26, 2004.

  This implementation supports Rouge-N where N is an int in [1, 9], RougeL, and
  RougeLsum. Note, to calculate multiple ROUGE Metrics, you will need to call
  this metric multiple times.

  For this implementation, a Label is expected to be a list of texts containing
  the target summaries. A Prediction is expected to be text containing the
  predicted text.

  In the ROUGE paper, two flavors of ROUGE are described:

  1. sentence-level: Compute longest common subsequence (LCS) between two pieces
  of text. Newlines are ignored. This is called 'rougeL' in this package.
  2. summary-level: Newlines in the text are interpreted as sentence boundaries,
  and the LCS is computed between each pair of reference and candidate
  sentences, and the union-LCS is computed. This is called
  'rougeLsum' in this package. This is the ROUGE-L reported in *[Get To The
  Point: Summarization with Pointer-Generator Networks]
  (https://arxiv.org/abs/1704.04368)*, for example. If your
  references/candidates do not have newline delimiters, you can use the
  split_summaries argument.

  This is a wrapper of the pure python implementation of ROUGE found here:
  https://pypi.org/project/rouge-score/

  To implement this metric, see the example below:

  eval_config = tfma.EvalConfig(
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='Rouge',
                  config='"rouge_type":"rouge1"'
          ]),
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='Rouge',
                  config='"rouge_type":"rouge2"'
          ]),
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='Rouge',
                  config='"rouge_type":"rougeL"'
          ]),
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='Rouge',
                  config='"rouge_type":"rougeLsum"'
          ]),
          ...
      ],
      ...
  )

  evaluator = tfx.borg.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    eval_config=eval_config)
  """

  def __init__(
      self,
      rouge_type: str,
      name: Optional[str] = None,
      use_stemmer: Optional[bool] = False,
      split_summaries: Optional[bool] = False,
      tokenizer: Optional[tokenizers.Tokenizer] = None,
  ):
    """Initializes ROUGE Metrics."""

    super().__init__(
        metric_util.merge_per_key_computations(_rouge),
        rouge_type=rouge_type,
        name=name or rouge_type,
        use_stemmer=use_stemmer,
        split_summaries=split_summaries,
        tokenizer=tokenizer,
    )


metric_types.register_metric(Rouge)
