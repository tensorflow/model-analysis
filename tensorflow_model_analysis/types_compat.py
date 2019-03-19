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
"""Types for backwards compatibility with versions that don't support typing."""

import collections
import sys

from apache_beam.typehints import Any, Dict, List, Tuple, Union, Optional, Iterable, Generator  # pylint: disable=unused-import,g-multiple-import

# pylint: disable=invalid-name
Callable = None
Generic = None
Sequence = None
Type = None
# TODO(xinzha): figure out whether we can use six.string_types in beam
AnyStr = Any
Text = Any

# pylint: enable=invalid-name


def NamedTuple(name, fields):
  """Replacement NamedTuple function for Python 2 compatibility."""
  field_names = [field_name for field_name, _ in fields]
  cls = collections.namedtuple(name, field_names)
  # Update the module for the returned namedtuple to be that of the caller.
  #
  # This is to work around the issue where collections.namedtuple uses the name
  # of this module (i.e. types_compat) as the module for the namedtuple
  # returned, which breaks pickling and unpickling the NamedTuple because
  # pickle tries to find it as types_compat.mytuple rather than
  # caller_module.mytuple.
  cls.__module__ = sys._getframe(1).f_globals['__name__']  # pylint: disable=protected-access
  return cls
