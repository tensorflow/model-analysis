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
"""General utilities."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import inspect
import sys
import traceback
import six
from typing import List, Optional, Text

# Separator used when combining multiple layers of Extracts keys into a single
# string. Normally we would like to use '.' or '/' as a separator, but the
# output gets written to a table backed by a proto based schema which limits the
# characters that can be used to [a-zA-Z_].
KEY_SEPARATOR = '__'
# Suffix for keys representing the top k keys associated with a sparse item.
KEYS_SUFFIX = 'keys'
# Suffix for keys representing the top k values associated with a sparse item.
VALUES_SUFFIX = 'values'


def unique_key(key: Text,
               current_keys: List[Text],
               update_keys: Optional[bool] = False) -> Text:
  """Returns a unique key given a list of current keys.

  If the key exists in current_keys then a new key with _1, _2, ..., etc
  appended will be returned, otherwise the key will be returned as passed.

  Args:
    key: desired key name.
    current_keys: List of current key names.
    update_keys: True to append the new key to current_keys.
  """
  index = 1
  k = key
  while k in current_keys:
    k = '%s_%d' % (key, index)
    index += 1
  if update_keys:
    current_keys.append(k)
  return k


def compound_key(keys: List[Text], separator: Text = KEY_SEPARATOR) -> Text:
  """Returns a compound key based on a list of keys.

  Args:
    keys: Keys used to make up compound key.
    separator: Separator between keys. To ensure the keys can be parsed out of
      any compound key created, any use of a separator within a key will be
      replaced by two separators.
  """
  return separator.join([key.replace(separator, separator * 2) for key in keys])


def create_keys_key(key: Text) -> Text:
  """Creates secondary key representing the sparse keys associated with key."""
  return '_'.join([key, KEYS_SUFFIX])


def create_values_key(key: Text) -> Text:
  """Creates secondary key representing sparse values associated with key."""
  return '_'.join([key, VALUES_SUFFIX])


def reraise_augmented(exception: Exception, additional_message: Text) -> None:
  """Reraise a given exception with additional information.

  Based on _reraise_augmented in Apache Beam.

  Args:
    exception: Original exception.
    additional_message: Additional message to append to original exception's
      message.
  """
  # To emulate exception chaining (not available in Python 2).
  original_traceback = sys.exc_info()[2]
  try:
    # Attempt to construct the same kind of exception
    # with an augmented message.
    #
    # pytype: disable=attribute-error
    # PyType doesn't know that Exception has the args attribute.
    new_exception = type(exception)(
        exception.args[0] + ' additional message: ' + additional_message,
        *exception.args[1:])
    # pytype: enable=attribute-error
  except:  # pylint: disable=bare-except
    # If anything goes wrong, construct a RuntimeError whose message
    # records the original exception's type and message.
    new_exception = RuntimeError(
        traceback.format_exception_only(type(exception), exception)[-1].strip()
        + ' additional message: ' + additional_message)

  six.reraise(type(new_exception), new_exception, original_traceback)


def kwargs_only(fn):
  """Wraps function so that callers must call it using keyword-arguments only.

  Args:
    fn: fn to wrap.

  Returns:
    Wrapped function that may only be called using keyword-arguments.
  """

  if hasattr(inspect, 'getfullargspec'):
    # For Python 3
    args = inspect.getfullargspec(fn)
    varargs = args.varargs
    keywords = args.varkw
  else:
    # For Python 2
    args = inspect.getargspec(fn)
    varargs = args.varargs
    keywords = args.keywords
  if varargs is not None:
    raise TypeError('function to wrap should not have *args parameter')
  if keywords is not None:
    raise TypeError('function to wrap should not have **kwargs parameter')

  arg_list = args.args
  has_default = [False] * len(arg_list)
  default_values = [None] * len(arg_list)
  has_self = arg_list[0] == 'self'
  if args.defaults:
    has_default[-len(args.defaults):] = [True] * len(args.defaults)
    default_values[-len(args.defaults):] = args.defaults

  def wrapped_fn(*args, **kwargs):
    """Wrapped function."""
    if args:
      if not has_self or (has_self and len(args) != 1):
        raise TypeError('function %s must be called using keyword-arguments '
                        'only.' % fn.__name__)

    if has_self:
      if len(args) != 1:
        raise TypeError('function %s has self argument but not called with '
                        'exactly 1 positional argument' % fn.__name__)
      kwargs['self'] = args[0]

    kwargs_to_pass = {}
    for arg_name, arg_has_default, arg_default_value in zip(
        arg_list, has_default, default_values):
      if not arg_has_default and arg_name not in kwargs:
        raise TypeError('function %s must be called with %s specified' %
                        (fn.__name__, arg_name))
      kwargs_to_pass[arg_name] = kwargs.pop(arg_name, arg_default_value)

    if kwargs:
      raise TypeError('function %s called with extraneous kwargs: %s' %
                      (fn.__name__, kwargs.keys()))

    return fn(**kwargs_to_pass)

  return wrapped_fn
