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

import collections
import dataclasses
from typing import Iterable, Optional, Sequence

from absl import logging
import apache_beam as beam
import numpy as np
import sacrebleu.metrics as sacrebleu
from tensorflow_model_analysis.metrics import metric_types
from tensorflow_model_analysis.metrics import metric_util
from tensorflow_model_analysis.proto import config_pb2


_BLEU_NAME_DEFAULT = 'BLEU'


# TODO: b/287700355) - Add __slots__ to this dataclass.
@dataclasses.dataclass
class _RefInfo:
  ngrams: collections.Counter[dict[tuple[str], int]]  # n-grams and counts
  lens: list[int]  # lengths


def _find_closest_ref_len(hyp_len: int, ref_lens: list[int]) -> int:
  """Given a hypothesis length and a list of reference lengths, returns the closest reference length.

  Args:
    hyp_len: The hypothesis length.
    ref_lens: A list of reference lengths. The closest reference length.

  Returns:
    The closest reference length, or -1 if ref_lens is empty.
  """
  ref_lens_arr = np.array(ref_lens)
  return ref_lens_arr[np.argmin(abs(ref_lens_arr - hyp_len))]


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
    self.bleu_metric = sacrebleu.BLEU(**bleu_kwargs)

  def _preprocess_segment(self, sentence: str) -> str:
    """Given a sentence, lowercases (optionally) and tokenizes it."""
    if self.bleu_metric.lowercase:
      sentence = sentence.lower()
    return self.bleu_metric.tokenizer(sentence.rstrip())

  def _extract_reference_info(self, refs: Sequence[str]) -> _RefInfo:
    """Given a list of reference segments, extract the n-grams and reference lengths.

    The latter will be useful when comparing hypothesis and reference lengths.

    Args:
      refs: A sequence of strings.

    Returns:
      A _RefInfo() with reference ngrams and lengths.
    """
    refs = iter(refs)

    final_ngrams, ref_len = sacrebleu.helpers.extract_all_word_ngrams(
        next(refs), 1, self.bleu_metric.max_ngram_order
    )
    ref_lens = [ref_len]

    for ref in refs:
      # Extract n-grams for this ref.
      new_ngrams, ref_len = sacrebleu.helpers.extract_all_word_ngrams(
          ref, 1, self.bleu_metric.max_ngram_order
      )

      ref_lens.append(ref_len)

      # Merge counts across multiple references.
      # The below loop is faster than 'final_ngrams |= new_ngrams'.
      for ngram, count in new_ngrams.items():
        final_ngrams[ngram] = max(final_ngrams[ngram], count)

    return _RefInfo(ngrams=final_ngrams, lens=ref_lens)

  def _extract_reference_ngrams_and_lens(
      self, references: Sequence[Sequence[str]]
  ) -> list[_RefInfo]:
    """Given the full set of document references, extract segment n-grams and lens."""
    ref_data = []

    # Iterate through all references.
    for refs in zip(*references):
      # Remove undefined references and seperate ngrams.
      lines = [
          self._preprocess_segment(line) for line in refs if line is not None
      ]

      # Get n-grams data.
      ref_data.append(self._extract_reference_info(lines))

    return ref_data

  def _compute_segment_statistics(
      self,
      hypothesis: str,
      ref_info: _RefInfo,
  ) -> list[int]:
    """Given a (pre-processed) hypothesis sentence and already computed reference n-grams & lengths, returns the best match statistics across the references.

    Args:
      hypothesis: Hypothesis sentence.
      ref_info: _RefInfo containing the counter with all n-grams and counts, and
        the list of reference lengths.

    Returns:
      A list of integers with match statistics.
    """
    # Extract n-grams for the hypothesis.
    hyp_ngrams, hyp_len = sacrebleu.helpers.extract_all_word_ngrams(
        hypothesis, 1, self.bleu_metric.max_ngram_order
    )

    ref_len = _find_closest_ref_len(hyp_len, ref_info.lens)

    # Count the stats.
    # Although counter has its internal & and | operators, this is faster.
    matching_ngrams = [0] * self.bleu_metric.max_ngram_order
    total_ngrams = matching_ngrams[:]

    for hyp_ngram, hyp_count in hyp_ngrams.items():
      # n-gram order.
      n = len(hyp_ngram) - 1

      # Count hypothesis n-grams.
      total_ngrams[n] += hyp_count

      # Count matched n-grams.
      ref_ngrams = ref_info.ngrams
      if hyp_ngram in ref_ngrams:
        matching_ngrams[n] += min(hyp_count, ref_ngrams[hyp_ngram])

    # Return a flattened list as per 'stats' semantics.
    return [hyp_len, ref_len] + matching_ngrams + total_ngrams

  def _extract_corpus_statistics(
      self,
      hypotheses: Sequence[str],
      references: Optional[Sequence[Sequence[str]]],
  ) -> list[list[int]]:
    """Reads the corpus and returns sentence-level match statistics for faster re-computations esp during statistical tests.

    Args:
      hypotheses: A sequence of hypothesis strings.
      references: A sequence of reference documents with document being defined
        as a sequence of reference strings of shape (batch_size_of_references x
        batch_size_of_hypotheses).

    Returns:
      A list where each sublist corresponds to segment statistics.
    """
    stats = []
    tok_count = 0

    # Extract the new 'stats'.
    for hyp, ref_kwargs in zip(
        hypotheses, self._extract_reference_ngrams_and_lens(references)
    ):
      # Check for already-tokenized input problem.
      if not self.bleu_metric._force and hyp.endswith(' .'):  # pylint:disable=protected-access
        tok_count += 1

      # Collect stats.
      stats.append(
          self._compute_segment_statistics(
              self._preprocess_segment(hyp), ref_kwargs
          )
      )

    if tok_count >= 100:
      logging.warning("That's 100 lines that end in a tokenized period (' .')")
      logging.warning(
          'It looks like you forgot to detokenize your test data, which may'
          ' hurt your score.'
      )
      logging.warning(
          "If you insist your data is detokenized, or don't care, you can"
          " suppress this message with the 'force' parameter."
      )

    return stats

  def _compute_score_from_stats(self, stats: list[int]) -> sacrebleu.BLEUScore:
    """Computes the final score from already aggregated statistics.

    'stats' semantics are preserved here from the wrapped implementation.
    stats = [hyp_len, ref_len, matching_ngrams, total_ngrams] where
      hyp_len = number of unigrams (words) in the hypothesis
      ref_len = number of unigrams (words) in the reference
        Note, ending punctuation (periods, exclamation points, etc.) count as
        their own unigram.
        For example, 'Google.' has 2 unigrams: 'Google' and '.'
      matching_ngrams[n - 1] = number of matching n-grams for n > 0
        matching_ngrams[0] = number of matching unigrams
        matching_ngrams[1] = number of matching bigrams
        ...
      total_ngrams[n - 1] = number of n-grams in hyp for n > 0
        total_ngrams[] follows same pattern as matching_ngrams[]

    Args:
      stats: A list of segment-level statistics.

    Returns:
      A 'BLEUScore' object.
    """
    bleu_metric = self.bleu_metric

    # matching_ngrams[n - 1] = number of matching n-grams for n > 0
    matching_ngrams = stats[2 : 2 + bleu_metric.max_ngram_order]

    # total_ngrams[n - 1] = number of n-grams in hyp for n > 0
    total_ngrams = stats[2 + bleu_metric.max_ngram_order :]

    # hyp_len = number of unigrams (words) in the hypothesis
    hyp_len = int(stats[0])

    # ref_len = number of unigrams (words) in the reference
    ref_len = int(stats[1])

    return self.bleu_metric.compute_bleu(
        correct=matching_ngrams,
        total=total_ngrams,
        sys_len=hyp_len,
        ref_len=ref_len,
        smooth_method=bleu_metric.smooth_method,
        smooth_value=bleu_metric.smooth_value,
        effective_order=bleu_metric.effective_order,
        max_ngram_order=bleu_metric.max_ngram_order,
    )

  def create_accumulator(self):
    """Accumulator is the running total of 'stats' of type np.ndarray.

    Args: None.

    Returns:
      'stats' list of all zeros.
    """
    # TODO: b/321082946 - Replace 'stats' semantics with a dataclass.
    # len(stats)
    # = len(hyp_len) + len(ref_len) + len(matching_ngrams) + len(total_ngrams)
    # = 1 + 1 + max_ngram_order + max_ngram_order = 2 + 2 * max_ngram_order
    return np.zeros(2 + 2 * self.bleu_metric.max_ngram_order, dtype=int)

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
            example_weighted=False,  # Example weights not honored.
            flatten=False,
            squeeze=False,
        )
    )

    # Sum accumulator and new stats
    return accumulator + np.sum(
        self._extract_corpus_statistics(hypotheses, references), axis=0
    )

  def merge_accumulators(
      self, list_of_stats: Iterable[np.ndarray]
  ) -> np.ndarray:
    """Sum of list of stats."""
    return np.sum(list_of_stats, axis=0)

  def extract_output(
      self, accumulator: np.ndarray
  ) -> dict[metric_types.MetricKey, sacrebleu.BLEUScore]:
    # TODO: b/319702245 - Resolve the issue below in compute_bleu().
    # We need to convert the accumulator to a list here.
    # If we leave it as a np.ndarray of ints, then sacrebleu will not be able to
    # add decimal smooth values to the stats list within compute_bleu().
    # If we convert it to an np.ndarray of floats, then sacrebleu will not be
    # able to propely set BLEUScore._verbose because there is no format code 'd'
    # for floats.
    return {self.key: self._compute_score_from_stats(accumulator.tolist())}


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
