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

suite('fairness-metrics-board tests', () => {
  let fairness;

  const SLICES = [
    'sex:male', 'Overall', 'sex:female', 'passed:yes', 'passed:no', 'year:9',
    'year:10', 'year:11', 'year:12', 'group:A', 'group:B'
  ];

  const SLICES_SORTED = [
    'Overall', 'group:A', 'group:B', 'passed:no', 'passed:yes', 'sex:female',
    'sex:male', 'year:10', 'year:11', 'year:12', 'year:9'
  ];

  function generate_bounded_value_data(slice) {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': {
        'totalWeightedExamples': 100,
        'accuracy': {
          'lowerBound': 0.3,
          'upperBound': 0.5,
          'value': 0.4,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.30': {
          'lowerBound': 0.2,
          'upperBound': 0.4,
          'value': 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.50': {
          'lowerBound': 0.2,
          'upperBound': 0.4,
          'value': 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.70': {
          'lowerBound': 0.2,
          'upperBound': 0.4,
          'value': 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
      }
    };
  }

  const BOUNDED_VALUE_DATA = SLICES.map(generate_bounded_value_data);
  const BOUNDED_VALUE_DATA_SORTED =
      SLICES_SORTED.map(generate_bounded_value_data);

  const BOUNDED_VALUE_DATA_WITH_OMITTED_SLICE = Array.from(BOUNDED_VALUE_DATA);
  BOUNDED_VALUE_DATA_WITH_OMITTED_SLICE.push({
    'slice': 'year:13',
    'sliceValue': '13',
    'metrics': {'__ERROR__': 'error message'}
  });

  test('checkMetrics', done => {
    fairness = fixture('test-fixture');

    const fillData = () => {
      fairness.thresholds = ['0.30', '0.50', '0.70'];
      fairness.data = BOUNDED_VALUE_DATA;
      fairness.weightColumn = 'totalWeightedExamples';
      fairness.metrics = ['post_export_metrics/false_negative_rate'];
      fairness.showFullTable_ = true;
      setTimeout(checkMetricsValue, 1);
    };

    const checkMetricsValue = () => {
      let metricSummary =
          fairness.shadowRoot.querySelector('fairness-metric-summary');

      assert.deepEqual(metricSummary.data, BOUNDED_VALUE_DATA);
      assert.deepEqual(
          metricSummary.metric, 'post_export_metrics/false_negative_rate');
      assert.deepEqual(
          metricSummary.slices, BOUNDED_VALUE_DATA_SORTED.map(v => v['slice']));
      assert.deepEqual(metricSummary.baseline, 'Overall');
      assert.deepEqual(metricSummary.thresholds, ['0.50']);

      done();
    };

    setTimeout(fillData, 0);
  });

  // Todo: Add unit test of the privacy container after the privacy component is
  // added.
});
