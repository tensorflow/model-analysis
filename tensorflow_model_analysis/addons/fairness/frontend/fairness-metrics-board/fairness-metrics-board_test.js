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

  function generateBoundedValueData(slice) {
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

  const BOUNDED_VALUE_DATA = SLICES.map(generateBoundedValueData);
  const BOUNDED_VALUE_DATA_SORTED = SLICES_SORTED.map(generateBoundedValueData);
  const BOUNDED_VALUE_DATA2 = SLICES.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': {
        'totalWeightedExamples': 100,
        'accuracy': {
          'lowerBound': 0.2,
          'upperBound': 0.4,
          'value': 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.30': {
          'lowerBound': 0.3,
          'upperBound': 0.4,
          'value': 0.35,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.50': {
          'lowerBound': 0.5,
          'upperBound': 0.5,
          'value': 0.5,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.70': {
          'lowerBound': 0.2,
          'upperBound': 0.4,
          'value': 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        '__ERROR__': 'random error message',
      }
    };
  });

  const BOUNDED_VALUE_DATA_WITH_OMITTED_SLICE = Array.from(BOUNDED_VALUE_DATA);
  BOUNDED_VALUE_DATA_WITH_OMITTED_SLICE.push({
    'slice': 'year:13',
    'sliceValue': '13',
    'metrics': {
      // Example count for this slice key is lower than the minimum required
      // value: 10. No data is aggregated
      '__ERROR__':
          'RXhhbXBsZSBjb3VudCBmb3IgdGhpcyBzbGljZSBrZXkgaXMgbG93ZXIgdGhhbiB0aGUgbWluaW11\nbSByZXF1aXJlZCB2YWx1ZTogMTAuIE5vIGRhdGEgaXMgYWdncmVnYXRlZA==',
    }
  });
  BOUNDED_VALUE_DATA_WITH_OMITTED_SLICE.push({
    'slice': 'year:14',
    'sliceValue': '14',
    'metrics': {
      '__ERROR__':
          'Example count for this slice key is lower than the minimum' +
          ' required value: 30. No data is aggregated for this slice.',
    }
  });

  test('checkMetrics', done => {
    fairness = fixture('test-fixture');

    const fillData = () => {
      fairness.data = BOUNDED_VALUE_DATA;
      fairness.weightColumn = 'totalWeightedExamples';
      fairness.metrics = ['post_export_metrics/false_negative_rate'];
      fairness.thresholdedMetrics =
          new Set(['post_export_metrics/false_negative_rate']);
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

      setTimeout(checkPrivacyContainer, 1);
    };

    const checkPrivacyContainer = () => {
      const privacyContainer =
          fairness.shadowRoot.querySelector('fairness-privacy-container');

      assert.deepEqual(privacyContainer.hidden, true);
      done();
    };

    setTimeout(fillData, 0);
  });

  test('checkMetrics_evalCompare', done => {
    fairness = fixture('test-fixture');

    const fillData = () => {
      fairness.data = BOUNDED_VALUE_DATA;
      fairness.evalName = 'EvalA';
      fairness.dataCompare = BOUNDED_VALUE_DATA2;
      fairness.evalNameCompare = 'EvalB';
      fairness.weightColumn = 'totalWeightedExamples';
      fairness.metrics = ['post_export_metrics/false_negative_rate'];
      fairness.thresholdedMetrics =
          new Set(['post_export_metrics/false_negative_rate']);
      fairness.showFullTable_ = true;
      setTimeout(checkMetricsValue, 1);
    };

    const checkMetricsValue = () => {

      let metricSummary =
          fairness.shadowRoot.querySelector('fairness-metric-summary');

      assert.deepEqual(metricSummary.data, BOUNDED_VALUE_DATA);
      assert.deepEqual(metricSummary.evalName, 'EvalA');
      assert.deepEqual(metricSummary.evalNameCompare, 'EvalB');
      assert.deepEqual(metricSummary.dataCompare, BOUNDED_VALUE_DATA2);
      assert.deepEqual(
          metricSummary.metric, 'post_export_metrics/false_negative_rate');
      assert.deepEqual(
          metricSummary.slices, BOUNDED_VALUE_DATA_SORTED.map(v => v['slice']));
      assert.deepEqual(metricSummary.baseline, 'Overall');

      done();
    };

    setTimeout(fillData, 0);
  });

  test('checkOmittedSliceErrorMessage', done => {
    fairness = fixture('test-fixture');

    const fillData = () => {
      fairness.data = BOUNDED_VALUE_DATA_WITH_OMITTED_SLICE;
      fairness.weightColumn = 'totalWeightedExamples';
      fairness.metrics = ['post_export_metrics/false_negative_rate'];
      fairness.thresholdedMetrics =
          new Set(['post_export_metrics/false_negative_rate']);
      fairness.showFullTable_ = true;
      setTimeout(checkPrivacyContainer, 1);
    };

    const checkPrivacyContainer = () => {
      const privacyContainer =
          fairness.shadowRoot.querySelector('fairness-privacy-container');

      assert.deepEqual(privacyContainer.hidden, false);
      assert.deepEqual(privacyContainer.omittedSlices, ['year:13', 'year:14']);
      done();
    };

    setTimeout(fillData, 0);
  });
});
