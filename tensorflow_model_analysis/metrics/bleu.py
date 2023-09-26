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
"""BLEU (Bilingual Evaluation Understudy) Metric.

http://aclweb.org/anthology/W14-3346
"""

from typing import Dict, Iterable, Optional

import apache_beam as beam
import numpy as np
import sacrebleu
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2


_BLEU_NAME_DEFAULT = 'BLEU'


class _BleuCombiner(beam.CombineFn):
  """Computes BLEU Score."""

  def __init__(
      self,
      eval_config: config_pb2.EvalConfig,
      model_name: str,
      output_name: str,
      key: metric_types.MetricKey,
      **bleu_kwargs,
  ):
    """Initializes BLEU Combiner.

    Args:
      eval_config: Eval config.
      model_name: The model for which to compute these metrics.
      output_name: The output name for which to compute these metrics.
      key: MetricKey for extract_output().
      **bleu_kwargs: kwargs to initialize BLEU Metric. Possible options include
        'lowercase' (If True, lowercases the input, enabling
        case-insensitivity.), 'force' (If True, insists that the tokenized input
        is detokenized.), 'tokenize' (Tokenization method to use for BLEU.
        Possible values are 'none' (No tokenization), 'zh' (Chinese
        tokenization), '13a' (mimics the mteval-v13a from Moses), and 'intl'
        (International tokenization, mimics the mteval-v14 script from Moses).),
        'smooth_method' (The smoothing method to use. Possible values are 'none'
        (no smoothing), 'floor' (increment zero counts), 'add-k' (increment
        num/denom by k for n>1), and 'exp' (exponential decay).), 'smooth_value'
        (The smoothing value. Only valid when smoothmethod='floor' or
        smooth_method='add-k'.), and 'effective_order' (If True, stops including
        n-gram orders for which precision is 0. This should be True if
        sentence-level BLEU will be computed.).
    """
    self.eval_config = eval_config
    self.model_name = model_name
    self.output_name = output_name
    self.key = key
    self.bleu_metric = sacrebleu.metrics.BLEU(**bleu_kwargs)
    # len(stats) = len(hyp_len) + len(ref_len) + len(correct) + len(total)
    # = 1 + 1 + max_ngram_order + max_ngram_order = 2 + 2 * max_ngram_order
    self.stats_len = 2 + 2 * self.bleu_metric.max_ngram_order

  def create_accumulator(self):
    """Accumulator is the running total of 'stats' of type np.ndarray.

    'stats' semantics are preserved here from the wrapped implementation.
    stats = [hyp_len, ref_len, correct, total] where
    hyp_len = number of unigrams (words) in the hypothesis
    ref_len = number of unigrams (words) in the reference
      Note, ending punctuation (periods, exclamation points, etc.) count as
      their own unigram.
      For example, 'Google.' has 2 unigrams: 'Google' and '.'
    correct[n - 1] = number of matching n-grams for n > 0
      correct[0] = number of matching unigrams
      correct[1] = number of matching bigrams
      ...
    total[n - 1] = (
        max(number of n-grams in hyp, number of n-grams in ref) for n > 0
    )
      total[] follows same pattern as correct[]

    Args: None.

    Returns:
      'stats' list of all zeros.
    """
    return np.zeros(self.stats_len, dtype=int)

  def add_input(
      self,
      accumulator: np.ndarray,
      element: metric_types.StandardMetricInputs,
  ) -> np.ndarray:
    # references = labels, hypotheses = predictions
    references, hypotheses, _ = next(
        metric_util.to_label_prediction_example_weight(
            element,
            eval_config=self.eval_config,
            model_name=self.model_name,
            output_name=self.output_name,
            example_weighted=False,  # Example weights not honored
            flatten=False,
            squeeze=False,
        )
    )

    # TODO(b/299345719): Remove call to protected member
    new_stats = self.bleu_metric._extract_corpus_statistics(  # pylint:disable=protected-access
        hypotheses.tolist(), references.tolist()
    )

    # Sum accumulator and new stats
    return accumulator + np.sum(new_stats, axis=0)

  def merge_accumulators(
      self, list_of_stats: Iterable[np.ndarray]
  ) -> np.ndarray:
    """Sum of list of stats."""
    return np.sum(list_of_stats, axis=0)

  def extract_output(
      self, accumulator: np.ndarray
  ) -> Dict[metric_types.MetricKey, sacrebleu.metrics.BLEUScore]:
    # TODO(b/299345719): Remove call to protected member
    return {
        self.key: self.bleu_metric._compute_score_from_stats(  # pylint:disable=protected-access
            accumulator.tolist()
        )
    }


def _bleu(
    name: str,
    eval_config: config_pb2.EvalConfig,
    model_name: str,
    output_name: str,
    lowercase: bool,
    force: bool,
    tokenize: str,
    smooth_method: str,
    smooth_value: float,
    effective_order: bool,
) -> metric_types.MetricComputations:
  """Returns BLEU score."""
  key = metric_types.MetricKey(name=name)
  return [
      metric_types.MetricComputation(
          keys=[key],
          preprocessors=None,
          combiner=_BleuCombiner(
              eval_config=eval_config,
              model_name=model_name,
              output_name=output_name,
              lowercase=lowercase,
              force=force,
              tokenize=tokenize,
              smooth_method=smooth_method,
              smooth_value=smooth_value,
              effective_order=effective_order,
              key=key,
          ),
      )
  ]


class Bleu(metric_types.Metric):
  """BLEU Metric."""

  def __init__(
      self,
      name: Optional[str] = _BLEU_NAME_DEFAULT,
      lowercase: Optional[bool] = False,
      force: Optional[bool] = False,
      tokenize: Optional[str] = '13a',
      smooth_method: Optional[str] = 'exp',
      smooth_value: Optional[float] = None,
      use_effective_order: Optional[bool] = False,
  ):
    """Initializes BLEU Metric."""

    # This is 'use_effective_order' and not 'effective_order' for backward
    # compatibility for old style BLEU API access (< 1.4.11)
    super().__init__(
        metric_util.merge_per_key_computations(_bleu),
        name=name,
        lowercase=lowercase,
        force=force,
        tokenize=tokenize,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=use_effective_order,
    )


metric_types.register_metric(Bleu)
