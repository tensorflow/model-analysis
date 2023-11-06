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
"""Lazytf core library."""


import abc


class CallableCombineFn(abc.ABC):
  """An aggregation interface, similar to apche_beam.CombineFn."""

  @abc.abstractmethod
  def create_accumulator(self):
    """Creates the initial states for the aggregation."""

  @abc.abstractmethod
  def add_input(self, accumulator, inputs):
    """Update the accumulator from a batch of inputs.

    Args:
      accumulator: the current accumulator.
      inputs: the elements to add, should not be mutated
    """

  @abc.abstractmethod
  def merge_accumulators(self, accumulators):
    """Mering multiple accumulators into a one accumulator value.

    Args:
      accumulators: the accumulators to be merged.
    """

  @abc.abstractmethod
  def extract_output(self, accumulator):
    """Computes and returns the result from accumulator.

    Args:
      accumulator: the final accumulator value computed by this CombineFn.
    """

  def __call__(self, inputs):
    """Directly apply aggregate on inputs."""
    return self.extract_output(
        self.add_input(self.create_accumulator(), inputs)
    )
