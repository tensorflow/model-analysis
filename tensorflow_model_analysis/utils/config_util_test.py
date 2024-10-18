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
"""Tests for config."""

import tensorflow as tf
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import config_util

from google.protobuf import text_format


class ConfigTest(tf.test.TestCase):

  def testVerifyEvalConfigMetricSpecPerSliceThresholdsNoSlicingSpec(self):
    eval_config_pbtxt = """
      model_specs { name: "" }
      metrics_specs {
        per_slice_thresholds {
          key: 'ExampleCount'
          value: {
            thresholds: {
              # missing slicing_specs
              threshold {
                value_threshold {
                  lower_bound { value: 0.9 }
                }
              }
            }
          }
        }
        metrics {
          class_name: "ExampleCount"
        }
      }
    """

    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())
    with self.assertRaisesRegex(ValueError, r'slicing_specs must be.*'):
      config_util.verify_eval_config(eval_config)

  def testVerifyEvalConfigMetricConfigPerSliceThresholdsNoSlicingSpec(self):
    eval_config_pbtxt = """
      model_specs { name: "" }
      metrics_specs {
        metrics {
          class_name: "ExampleCount"
          per_slice_thresholds {
            # missing slicing_specs
            threshold {
              value_threshold {
                lower_bound { value: 0.9 }
              }
            }
          }
        }
      }
    """

    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())
    with self.assertRaisesRegex(ValueError, r'slicing_specs must be.*'):
      config_util.verify_eval_config(eval_config)

  def testUpdateConfigWithDefaultsNoBaselineModelNonRubberstamp(self):
    eval_config_pbtxt = """
      model_specs { name: "" }
      metrics_specs {
        metrics {
          class_name: "MeanLabel"
          per_slice_thresholds {
            slicing_specs: {}
            threshold {
              value_threshold {
                lower_bound { value: 0.9 }
              }
              change_threshold {
                direction: HIGHER_IS_BETTER
                absolute { value: -1e-10 }
              }
            }
          }
        }
      }
    """

    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    with self.assertRaises(RuntimeError):
      config_util.update_eval_config_with_defaults(
          eval_config, has_baseline=False, rubber_stamp=False)

  def testUpdateConfigWithDefaultsNoModel(self):
    eval_config_pbtxt = """
      metrics_specs {
        metrics { class_name: "ExampleCount" }
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs { name: "" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: [""]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    got_eval_config = config_util.update_eval_config_with_defaults(eval_config)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testUpdateConfigWithDefaultsEmtpyModelName(self):
    eval_config_pbtxt = """
      model_specs { name: "" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs { name: "" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: [""]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    got_eval_config = config_util.update_eval_config_with_defaults(eval_config)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testUpdateConfigWithDefaultsSingleModel(self):
    eval_config_pbtxt = """
      model_specs { name: "model1" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
      }
      metrics_specs {
        metrics { class_name: "MeanLabel" }
        model_names: ["model1"]
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs { name: "" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: [""]
      }
      metrics_specs {
        metrics { class_name: "MeanLabel" }
        model_names: [""]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    got_eval_config = config_util.update_eval_config_with_defaults(eval_config)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testUpdateConfigWithDefaultsMultiModel(self):
    eval_config_pbtxt = """
      model_specs { name: "model1" }
      model_specs { name: "model2" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
      }
      metrics_specs {
        metrics { class_name: "MeanLabel" }
        model_names: ["model1"]
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs { name: "model1" }
      model_specs { name: "model2" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: ["model1", "model2"]
      }
      metrics_specs {
        metrics { class_name: "MeanLabel" }
        model_names: ["model1"]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    got_eval_config = config_util.update_eval_config_with_defaults(eval_config)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testUpdateConfigWithDefaultsBaselineModel(self):
    eval_config_pbtxt = """
      model_specs { name: "candidate" }
      model_specs { name: "baseline" is_baseline: true }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs { name: "candidate" }
      model_specs { name: "baseline" is_baseline: true }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: ["candidate", "baseline"]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    got_eval_config = config_util.update_eval_config_with_defaults(
        eval_config, has_baseline=True)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testUpdateConfigWithoutBaselineModelWhenModelNameProvided(self):
    eval_config_pbtxt = """
      model_specs { name: "candidate" }
      model_specs { name: "baseline" is_baseline: true }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: "candidate"
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs { name: "candidate" }
      model_specs { name: "baseline" is_baseline: true }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: ["candidate"]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    got_eval_config = config_util.update_eval_config_with_defaults(
        eval_config, has_baseline=True)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testUpdateConfigWithDefaultsAutomaticallyAddsBaselineModel(self):
    eval_config_pbtxt = """
      model_specs { label_key: "my_label" }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs { name: "candidate" label_key: "my_label" }
      model_specs { name: "baseline" label_key: "my_label" is_baseline: true }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: ["candidate", "baseline"]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    got_eval_config = config_util.update_eval_config_with_defaults(
        eval_config, has_baseline=True)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testUpdateConfigWithDefaultsDoesNotAutomaticallyAddBaselineModel(self):
    eval_config_pbtxt = """
      model_specs { name: "model1" }
      model_specs { name: "model2" is_baseline: true }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs { name: "model1" }
      model_specs { name: "model2" is_baseline: true }
      metrics_specs {
        metrics { class_name: "ExampleCount" }
        model_names: ["model1", "model2"]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    got_eval_config = config_util.update_eval_config_with_defaults(
        eval_config, has_baseline=True)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testUpdateConfigWithDefaultsRemoveBaselineModel(self):
    eval_config_pbtxt = """
      model_specs { name: "candidate" }
      model_specs { name: "baseline" is_baseline: true }
      metrics_specs {
        metrics {
          class_name: "MeanLabel"
          threshold {
            value_threshold {
              lower_bound { value: 0.9 }
            }
            change_threshold {
              direction: HIGHER_IS_BETTER
              absolute { value: -1e-10 }
            }
          }
          per_slice_thresholds {
            threshold {
              value_threshold {
                lower_bound { value: 0.9 }
              }
              change_threshold {
                direction: HIGHER_IS_BETTER
                absolute { value: -1e-10 }
              }
            }
          }
          cross_slice_thresholds {
            threshold {
              value_threshold {
                lower_bound { value: 0.9 }
              }
              change_threshold {
                direction: HIGHER_IS_BETTER
                absolute { value: -1e-10 }
              }
            }
          }
        }
        thresholds {
          key: "my_metric"
          value {
            value_threshold {
              lower_bound { value: 0.9 }
            }
            change_threshold {
              direction: HIGHER_IS_BETTER
              absolute { value: -1e-10 }
            }
          }
        }
        per_slice_thresholds {
          key: "my_metric"
          value {
            thresholds {
              threshold {
                value_threshold {
                  lower_bound { value: 0.9 }
                }
                change_threshold {
                  direction: HIGHER_IS_BETTER
                  absolute { value: -1e-10 }
                }
              }
            }
          }
        }
        cross_slice_thresholds {
          key: "my_metric"
          value {
            thresholds {
              threshold {
                value_threshold {
                  lower_bound { value: 0.9 }
                }
                change_threshold {
                  direction: HIGHER_IS_BETTER
                  absolute { value: -1e-10 }
                }
              }
            }
          }
        }
      }
    """
    eval_config = text_format.Parse(eval_config_pbtxt, config_pb2.EvalConfig())

    expected_eval_config_pbtxt = """
      model_specs {}
      metrics_specs {
        metrics {
          class_name: "MeanLabel"
          threshold {
            value_threshold {
              lower_bound { value: 0.9 }
            }
          }
          per_slice_thresholds {
            threshold {
              value_threshold {
                lower_bound { value: 0.9 }
              }
            }
          }
          cross_slice_thresholds {
            threshold {
              value_threshold {
                lower_bound { value: 0.9 }
              }
            }
          }
        }
        thresholds {
          key: "my_metric"
          value {
            value_threshold {
              lower_bound { value: 0.9 }
            }
          }
        }
        per_slice_thresholds {
          key: "my_metric"
          value {
            thresholds {
              threshold {
                value_threshold {
                  lower_bound { value: 0.9 }
                }
              }
            }
          }
        }
        cross_slice_thresholds {
          key: "my_metric"
          value {
            thresholds {
              threshold {
                value_threshold {
                  lower_bound { value: 0.9 }
                }
              }
            }
          }
        }
        model_names: [""]
      }
    """
    expected_eval_config = text_format.Parse(expected_eval_config_pbtxt,
                                             config_pb2.EvalConfig())

    # Only valid when rubber stamping.
    got_eval_config = config_util.update_eval_config_with_defaults(
        eval_config, has_baseline=False, rubber_stamp=True)
    self.assertProtoEquals(got_eval_config, expected_eval_config)

  def testHasChangeThreshold(self):
    eval_config = text_format.Parse(
        """
      metrics_specs {
        metrics {
          class_name: "MeanLabel"
          threshold {
            change_threshold {
              direction: HIGHER_IS_BETTER
              absolute { value: 0.1 }
            }
          }
        }
      }
    """, config_pb2.EvalConfig())

    self.assertTrue(config_util.has_change_threshold(eval_config))

    eval_config = text_format.Parse(
        """
      metrics_specs {
        thresholds {
          key: "my_metric"
          value {
            change_threshold {
              direction: HIGHER_IS_BETTER
              absolute { value: 0.1 }
            }
          }
        }
      }
    """, config_pb2.EvalConfig())

    self.assertTrue(config_util.has_change_threshold(eval_config))

    eval_config = text_format.Parse(
        """
      metrics_specs {
        metrics {
          class_name: "MeanLabel"
          threshold {
            value_threshold {
              lower_bound { value: 0.9 }
            }
          }
        }
      }
    """, config_pb2.EvalConfig())

    self.assertFalse(config_util.has_change_threshold(eval_config))


