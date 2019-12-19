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

  testRemovePostExportMetrics() {
    assertNotUndefined(Util.removePostExportMetrics);
    assertEquals('', Util.removePostExportMetrics(''));
    assertEquals('accuracy', Util.removePostExportMetrics('accuracy'));
    assertEquals(
        'false_positive_rate',
        Util.removePostExportMetrics(
            'post_export_metrics/false_positive_rate'));
    assertEquals(
        'post_export_metrics',
        Util.removePostExportMetrics(
            'post_export_metrics'));
    assertEquals(
        '',
        Util.removePostExportMetrics(
            'post_export_metrics/'));
    assertEquals(
        'post_export_metrics/false_positive_rate',
        Util.removePostExportMetrics(
            'post_export_metrics/post_export_metrics/false_positive_rate'));
    assertEquals(
        'post_export_metrics_foo/bar',
        Util.removePostExportMetrics(
            'post_export_metrics_foo/bar'));
  }
});
