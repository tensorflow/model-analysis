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

suite('fairness-metrics-table tests', () => {
  const DATA = [
    {
      'slice': 'col:1',
      'metrics': {
        'loss': 0.7,
        'fairness_indicators_metrics/false_negative_rate@0.5': 0.7,
        'example_count': 1,
      }
    },
    {
      'slice': 'col:2',
      'metrics': {
        'loss': 0.72,
        'fairness_indicators_metrics/false_negative_rate@0.5': 0.72,
        'example_count': 1,
      }
    },
    {
      'slice': 'col:3',
      'metrics': {
        'loss': 0.74,
        'fairness_indicators_metrics/false_negative_rate@0.5': 0.74,
        'example_count': 1,
      },
    }
  ];
  const DATA_TO_COMPARE = [
    {
      'slice': 'col:1',
      'metrics': {
        'loss': 0.35,
        'fairness_indicators_metrics/false_negative_rate@0.5': 0.35,
        'example_count': 1,
      }
    },
    {
      'slice': 'col:2',
      'metrics': {
        'loss': 0.36,
        'fairness_indicators_metrics/false_negative_rate@0.5': 0.36,
        'example_count': 1,
      }
    },
    {
      'slice': 'col:3',
      'metrics': {
        'loss': 0.37,
        'fairness_indicators_metrics/false_negative_rate@0.5': 0.37,
        'example_count': 1,
      },
    }
  ];
  const BOUNDED_DATA = [
    {
      'slice': 'col:1',
      'metrics': {
        'loss': {
          'lowerBound': 0.6,
          'upperBound': 0.8,
          'value': 0.7,
        },
        'fairness_indicators_metrics/false_negative_rate@0.5': {
          'lowerBound': 0.6,
          'upperBound': 0.8,
          'value': 0.7,
        },
        'example_count': {
          'lowerBound': 1,
          'upperBound': 1,
          'value': 1,
        },
      }
    },
    {
      'slice': 'col:2',
      'metrics': {
        'loss': {
          'lowerBound': 0.62,
          'upperBound': 0.82,
          'value': 0.72,
        },
        'fairness_indicators_metrics/false_negative_rate@0.5': {
          'lowerBound': 0.62,
          'upperBound': 0.82,
          'value': 0.72,
        },
        'example_count': {
          'lowerBound': 1,
          'upperBound': 1,
          'value': 1,
        },
      }
    },
    {
      'slice': 'col:3',
      'metrics': {
        'loss': {
          'lowerBound': 0.64,
          'upperBound': 0.84,
          'value': 0.74,
        },
        'fairness_indicators_metrics/false_negative_rate@0.5': {
          'lowerBound': 0.64,
          'upperBound': 0.84,
          'value': 0.74,
        },
        'example_count': {
          'lowerBound': 1,
          'upperBound': 1,
          'value': 1,
        },
      }
    }
  ];
  const BOUNDED_DATA_TO_COMPARE = [
    {
      'slice': 'col:1',
      'metrics': {
        'loss': {
          'lowerBound': 0.25,
          'upperBound': 0.45,
          'value': 0.35,
        },
        'fairness_indicators_metrics/false_negative_rate@0.5': {
          'lowerBound': 0.25,
          'upperBound': 0.45,
          'value': 0.35,
        },
        'example_count': {
          'lowerBound': 1,
          'upperBound': 1,
          'value': 1,
        },
      }
    },
    {
      'slice': 'col:2',
      'metrics': {
        'loss': {
          'lowerBound': 0.26,
          'upperBound': 0.46,
          'value': 0.36,
        },
        'fairness_indicators_metrics/false_negative_rate@0.5': {
          'lowerBound': 0.26,
          'upperBound': 0.46,
          'value': 0.36,
        },
        'example_count': {
          'lowerBound': 1,
          'upperBound': 1,
          'value': 1,
        },
      }
    },
    {
      'slice': 'col:3',
      'metrics': {
        'loss': {
          'lowerBound': 0.27,
          'upperBound': 0.47,
          'value': 0.37,
        },
        'fairness_indicators_metrics/false_negative_rate@0.5': {
          'lowerBound': 0.27,
          'upperBound': 0.47,
          'value': 0.37,
        },
        'example_count': {
          'lowerBound': 1,
          'upperBound': 1,
          'value': 1,
        },
      }
    }
  ];

  const MODEL_A_NAME = 'ModelA';
  const MODEL_B_NAME = 'ModelB';

  const TEST_STEP_TIMEOUT_MS = 100;

  const TEST_CASES = [
    {
      test_name: 'DoubleValues_GeneralMetrics_NonComparison',
      test_data: {
        data: DATA,
        dataCompare: undefined,
        evalName: undefined,
        evalNameCompare: undefined,
        baseline: 'col:1',
        slices: ['col:2', 'col:3'],
        metrics: ['loss']
      },
      expected_data: {
        expected_header:
            ['feature', 'loss', 'loss against col:1', 'example count'],
        expected_rows: [
          ['col:1', '0.7', '0%', 1],
          ['col:2', '0.72', '2.857%', 1],
          ['col:3', '0.74', '5.714%', 1],
        ]
      }
    },
    {
      test_name: 'BoundedValues_GeneralMetrics_NonComparison',
      test_data: {
        data: BOUNDED_DATA,
        dataCompare: undefined,
        evalName: undefined,
        evalNameCompare: undefined,
        baseline: 'col:1',
        slices: ['col:2', 'col:3'],
        metrics: ['loss']
      },
      expected_data: {
        expected_header:
            ['feature', 'loss', 'loss against col:1', 'example count'],
        expected_rows: [
          ['col:1', '0.7 (0.6, 0.8)', '0%', '1 (1, 1)'],
          ['col:2', '0.72 (0.62, 0.82)', '2.857%', '1 (1, 1)'],
          ['col:3', '0.74 (0.64, 0.84)', '5.714%', '1 (1, 1)'],
        ]
      }
    },
    {
      test_name: 'DoubleValues_ThresholdMetrics_NonComparison',
      test_data: {
        data: DATA,
        dataCompare: undefined,
        evalName: undefined,
        evalNameCompare: undefined,
        baseline: 'col:1',
        slices: ['col:2', 'col:3'],
        metrics: ['fairness_indicators_metrics/false_negative_rate@0.5'],
      },
      expected_data: {
        expected_header: [
          'feature', 'false_negative_rate@0.5',
          'false_negative_rate@0.5 against col:1', 'example count'
        ],
        expected_rows: [
          ['col:1', '0.7', '0%', 1],
          ['col:2', '0.72', '2.857%', 1],
          ['col:3', '0.74', '5.714%', 1],
        ]
      }
    },
    {
      test_name: 'BoundedValues_ThresholdMetrics_NonComparison',
      test_data: {
        data: BOUNDED_DATA,
        dataCompare: undefined,
        evalName: undefined,
        evalNameCompare: undefined,
        baseline: 'col:1',
        slices: ['col:2', 'col:3'],
        metrics: ['fairness_indicators_metrics/false_negative_rate@0.5'],
      },
      expected_data: {
        expected_header: [
          'feature', 'false_negative_rate@0.5',
          'false_negative_rate@0.5 against col:1', 'example count'
        ],
        expected_rows: [
          ['col:1', '0.7 (0.6, 0.8)', '0%', '1 (1, 1)'],
          ['col:2', '0.72 (0.62, 0.82)', '2.857%', '1 (1, 1)'],
          ['col:3', '0.74 (0.64, 0.84)', '5.714%', '1 (1, 1)'],
        ],
      }
    },
    {
      test_name: 'DoubleValues_ThresholdMetrics_Comparison',
      test_data: {
        data: DATA,
        dataCompare: DATA_TO_COMPARE,
        evalName: MODEL_A_NAME,
        evalNameCompare: MODEL_B_NAME,
        baseline: 'col:1',
        slices: ['col:2', 'col:3'],
        metrics: ['fairness_indicators_metrics/false_negative_rate@0.5'],
      },
      expected_data: {
        expected_header: [
          'feature', 'ModelA@0.5', 'ModelB@0.5', 'ModelA against ModelB @0.5',
          'example count'
        ],
        expected_rows: [
          ['col:1', '0.7', '0.35', '-50%', 1],
          ['col:2', '0.72', '0.36', '-50%', 1],
          ['col:3', '0.74', '0.37', '-50%', 1],
        ],
      }
    },
    {
      test_name: 'BoundedValues_ThresholdMetrics_Comparison',
      test_data: {
        data: BOUNDED_DATA,
        dataCompare: BOUNDED_DATA_TO_COMPARE,
        evalName: MODEL_A_NAME,
        evalNameCompare: MODEL_B_NAME,
        baseline: 'col:1',
        slices: ['col:2', 'col:3'],
        metrics: ['fairness_indicators_metrics/false_negative_rate@0.5'],
      },
      expected_data: {
        expected_header: [
          'feature', 'ModelA@0.5', 'ModelB@0.5', 'ModelA against ModelB @0.5',
          'example count'
        ],
        expected_rows: [
          ['col:1', '0.7 (0.6, 0.8)', '0.35 (0.25, 0.45)', '-50%', '1 (1, 1)'],
          [
            'col:2', '0.72 (0.62, 0.82)', '0.36 (0.26, 0.46)', '-50%',
            '1 (1, 1)'
          ],
          [
            'col:3', '0.74 (0.64, 0.84)', '0.37 (0.27, 0.47)', '-50%',
            '1 (1, 1)'
          ],
        ],
      }
    }
  ];

  // Returns a promise that resolves after a given amount of time.
  const delay = (delayInMs) => {
    return new Promise((resolve) => setTimeout(resolve, delayInMs));
  };

  for (const test_case of TEST_CASES) {
    test(test_case.test_name, async (done) => {
      const table = fixture('test-fixture');
      // Setup input of the compnent.
      table.data = test_case.test_data.data;
      table.dataCompare = test_case.test_data.dataCompare;
      table.evalName = test_case.test_data.evalName;
      table.evalNameCompare = test_case.test_data.evalNameCompare;
      table.baseline = test_case.test_data.baseline;
      table.slices = test_case.test_data.slices;
      table.metrics = test_case.test_data.metrics;
      await delay(TEST_STEP_TIMEOUT_MS);

      // Check the computation of table rows.
      assert.deepEqual(
          table.headerRow_, test_case.expected_data.expected_header);
      for (let i = 0; i < table.contentRows_.length; i++) {
        const row = table.contentRows_[i];
        for (let j = 0; j < row.length; j++) {
          assert.equal(
              row[j].text, test_case.expected_data.expected_rows[i][j]);
        }
      }
      done();
    });
  }

  test('formatCell', done => {
    const table = fixture('test-fixture');
    assert.deepEqual(table.formatCell_(undefined), {
      text: 'NO_DATA',
      arrow: undefined,
      arrow_icon_css_class: undefined,
    });
    assert.deepEqual(
        table.formatCell_({
          'value': 0.5,
          'lowerBound': 0.4,
          'upperBound': 0.6,
        }),
        {
          text: '0.5 (0.4, 0.6)',
          arrow: undefined,
          arrow_icon_css_class: undefined,
        });
    assert.deepEqual(table.formatCell_(0), {
      text: '0',
      arrow: undefined,
      arrow_icon_css_class: undefined,
    });
    assert.deepEqual(table.formatCell_(NaN), {
      text: 'NaN',
      arrow: undefined,
      arrow_icon_css_class: undefined,
    });
  });

  test('formatComparisonCell', done => {
    const table = fixture('test-fixture');
    assert.deepEqual(
        table.formatComparisonCell_(undefined, undefined, undefined), {
          text: 'NO_DATA',
          arrow: undefined,
          arrow_icon_css_class: undefined,
        });
    assert.deepEqual(
        table.formatComparisonCell_(
            {
              'value': 0.5,
              'lowerBound': 0.4,
              'upperBound': 0.6,
            },
            {
              'value': 0.5,
              'lowerBound': 0.4,
              'upperBound': 0.6,
            },
            'metric'),
        {
          text: '0%',
          arrow: '',
          arrow_icon_css_class: 'blue-icon',
        });
    assert.deepEqual(table.formatComparisonCell_(NaN, NaN, 'metric'), {
      text: 'NaN%',
      arrow: '',
      arrow_icon_css_class: 'blue-icon',
    });
    assert.deepEqual(table.formatComparisonCell_(1, 0, 'metric'), {
      text: 'Infinity%',
      arrow: 'arrow-upward',
      arrow_icon_css_class: 'blue-icon',
    });
  });

  test('ToPercentage', done => {
    const table = fixture('test-fixture');
    assert.equal(table.toPercentage_(0.25), '25%');
    assert.equal(table.toPercentage_('0.6'), '60%');
  });

  test('Arrow', done => {
    const table = fixture('test-fixture');
    assert.equal(table.arrow_(0.25), 'arrow-upward');
    assert.equal(table.arrow_('-1.6'), 'arrow-downward');
    assert.equal(table.arrow_(0), '');
    assert.equal(table.arrow_(''), '');
  });

  test('IconClass', done => {
    const table = fixture('test-fixture');

    assert.equal(
        table.arrowIconCssClass_(0.25, 'false_positive_rate'), 'red-icon');
    assert.equal(
        table.arrowIconCssClass_('-0.25', 'false_positive_rate'), 'green-icon');

    assert.equal(
        table.arrowIconCssClass_('0.25', 'true_positive_rate'), 'green-icon');
    assert.equal(
        table.arrowIconCssClass_(-0.25, 'true_positive_rate'), 'red-icon');

    assert.equal(
        table.arrowIconCssClass_('0.25', 'true_neutral_rate'), 'blue-icon');
    assert.equal(
        table.arrowIconCssClass_(-0.25, 'true_neutral_rate'), 'blue-icon');
  });
});
