/**
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

goog.module('tensorflow_model_analysis.addons.fairness.frontend.UtilTest');
goog.setTestOnly('tensorflow_model_analysis.addons.fairness.frontend.UtilTest');

const Util = goog.require('tensorflow_model_analysis.addons.fairness.frontend.Util');
const testSuite = goog.require('goog.testing.testSuite');

testSuite({
  testRemoveMetricNamePrefix() {
    assertNotUndefined(Util.removeMetricNamePrefix);
    assertEquals('', Util.removeMetricNamePrefix(''));
    assertEquals('accuracy', Util.removeMetricNamePrefix('accuracy'));
    assertEquals(
        'false_positive_rate',
        Util.removeMetricNamePrefix('post_export_metrics/false_positive_rate'));
    assertEquals(
        'false_positive_rate',
        Util.removeMetricNamePrefix(
            'fairness_indicators_metrics/false_positive_rate'));
    assertEquals(
        'post_export_metrics',
        Util.removeMetricNamePrefix('post_export_metrics'));
    assertEquals(
        'fairness_indicators_metrics',
        Util.removeMetricNamePrefix('fairness_indicators_metrics'));
    assertEquals('', Util.removeMetricNamePrefix('post_export_metrics/'));
    assertEquals(
        '', Util.removeMetricNamePrefix('fairness_indicators_metrics/'));
    assertEquals(
        'post_export_metrics/false_positive_rate',
        Util.removeMetricNamePrefix(
            'post_export_metrics/post_export_metrics/false_positive_rate'));
    assertEquals(
        'fairness_indicators_metrics/false_positive_rate',
        Util.removeMetricNamePrefix(
            'fairness_indicators_metrics/fairness_indicators_metrics/false_positive_rate'));
    assertEquals(
        'post_export_metrics_foo/bar',
        Util.removeMetricNamePrefix('post_export_metrics_foo/bar'));
  },

  testGetMetricsValues() {
    assertNotUndefined(Util.getMetricsValues);
    assertObjectEquals(Util.getMetricsValues({}, 'example_count'), {});
    assertEquals(
        Util.getMetricsValues(
            {'metrics': {'example_count': 1}}, 'example_count'),
        1);
    assertEquals(
        Util.getMetricsValues(
            {'metrics': {'post_export_metrics/example_count': 1}},
            'example_count'),
        1);
  },

  testExtractFairnessMetric() {
    assertNotUndefined(Util.extractFairnessMetric);
    const fairness_metric1 =
        Util.extractFairnessMetric('post_export_metrics/positive_rate@50');
    assertEquals('post_export_metrics/positive_rate', fairness_metric1.name);
    assertEquals('50', fairness_metric1.threshold);

    const fairness_metric2 = Util.extractFairnessMetric('negative_rate@0');
    assertEquals('negative_rate', fairness_metric2.name);
    assertEquals('0', fairness_metric2.threshold);

    const fairness_metric3 = Util.extractFairnessMetric('neutral_rate');
    assertNull('neutral_rate', fairness_metric3);

    const fairness_metric4 = Util.extractFairnessMetric('accuracy');
    assertNull('accuracy', fairness_metric4);

    const fairness_metric5 = Util.extractFairnessMetric('accuracy@0.5');
    assertEquals('accuracy', fairness_metric5.name);
    assertEquals('0.5', fairness_metric5.threshold);
  }
});
