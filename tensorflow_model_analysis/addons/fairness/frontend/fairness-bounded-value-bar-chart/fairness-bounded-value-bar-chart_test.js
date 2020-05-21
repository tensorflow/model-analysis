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

suite('fairness-bounded-value-bar-chart tests', () => {
  let barChart;

  const SLICES = [
    'Overall', 'Slice:1', 'Slice:2', 'Slice:3', 'Slice:4', 'Slice:5', 'Slice:6',
    'Slice:7', 'Slice:8', 'Slice:9'
  ];

  const FNR_AT_30_VALUES = {
    'Overall': 0.5,
    'Slice:1': 0.95,
    'Slice:2': 0.5,
    'Slice:3': 0.90,
    'Slice:4': 0.10,
    'Slice:5': 0.85,
    'Slice:6': 0.15,
    'Slice:7': 0.85,
    'Slice:8': 0.20,
    'Slice:9': 0.75,
    'Slice:10': 0.25,
    'Slice:11': 0.70,
    'Slice:12': 0.30,
    'Slice:13': 0.65,
    'Slice:14': 0.35,
    'Slice:15': 0.60,
    'Slice:16': 0.85,
  };

  const BOUNDED_VALUE_DATA = SLICES.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': {
        'accuracy': {
          'lowerBound': 0.3,
          'upperBound': 0.5,
          'value': 0.4,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.30': {
          'lowerBound': FNR_AT_30_VALUES[slice] - 0.1,
          'upperBound': FNR_AT_30_VALUES[slice] + 0.1,
          'value': FNR_AT_30_VALUES[slice],
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
        'post_export_metrics/example_count': Math.floor(Math.random() * 100),
      }
    };
  });

  const SORTED_BOUNDED_VALUE_DATA = BOUNDED_VALUE_DATA.slice(0, 1).concat(
      BOUNDED_VALUE_DATA.slice(1, BOUNDED_VALUE_DATA.length)
          .sort(function(a, b) {
            return FNR_AT_30_VALUES[a.slice] - FNR_AT_30_VALUES[b.slice];
          }));

  const BOUNDED_VALUE_DATA_EQUAL = SLICES.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': {
        'accuracy': {
          'lowerBound': 0.3,
          'upperBound': 0.5,
          'value': 0.4,
          'methodology': 'POISSON_BOOTSTRAP'
        },
        'post_export_metrics/false_negative_rate@0.30': {
          'lowerBound': 0.4,
          'upperBound': 0.6,
          'value': 0.5,
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
        'post_export_metrics/example_count': Math.floor(Math.random() * 100),
      }
    };
  });

  test('AxisCheckForBoundedValue', done => {
    barChart = fixture('main');

    const fillData = () => {
      barChart.data = BOUNDED_VALUE_DATA;
      barChart.metrics = [
        'post_export_metrics/false_negative_rate@0.30',
        'post_export_metrics/false_negative_rate@0.50'
      ];
      barChart.baseline = SLICES[0];
      barChart.slices = SLICES.slice(1);
      setTimeout(checkValue, 200);
    };

    const checkValue = () => {
      const svg = d3.select(barChart.shadowRoot.querySelector('svg'));
      svg.select('#xaxis').selectAll('g').text((d, i) => {
        assert.equal(d, SORTED_BOUNDED_VALUE_DATA[i]['sliceValue']);
        return d;
      });
      done();
    };
    setTimeout(fillData, 0);
  });

  test('AxisCheckForBoundedValueEqual', done => {
    barChart = fixture('main');

    const fillData = () => {
      barChart.data = BOUNDED_VALUE_DATA_EQUAL;
      barChart.metrics = [
        'post_export_metrics/false_negative_rate@0.30',
        'post_export_metrics/false_negative_rate@0.50'
      ];
      barChart.baseline = SLICES[0];
      barChart.slices = SLICES.slice(1);
      setTimeout(checkValue, 200);
    };

    const checkValue = () => {
      const svg = d3.select(barChart.shadowRoot.querySelector('svg'));
      svg.select('#xaxis').selectAll('g').text((d, i) => {
        assert.equal(d, BOUNDED_VALUE_DATA_EQUAL[i]['sliceValue']);
        return d;
      });
      done();
    };
    setTimeout(fillData, 0);
  });

  test('BarCheckForBoundedValue', done => {
    barChart = fixture('main');

    const fillData = () => {
      barChart.data = BOUNDED_VALUE_DATA;
      barChart.metrics = [
        'post_export_metrics/false_negative_rate@0.30',
        'post_export_metrics/false_negative_rate@0.50'
      ];
      barChart.baseline = SLICES[0];
      barChart.slices = SLICES.slice(1);
      setTimeout(checkValue, 100);
    };

    const checkValue = () => {
      assert.equal(
          d3.select(barChart.shadowRoot.querySelector('svg'))
              .select('#bars')
              .selectAll('g')
              .nodes()
              .length,
          barChart.slices.length + 1);
      assert.equal(
          d3.select(barChart.shadowRoot.querySelector('svg'))
              .select('#bars')
              .selectAll('rect')
              .nodes()
              .length,
          (barChart.slices.length + 1) * barChart.metrics.length);
      assert.equal(
          d3.select(barChart.shadowRoot.querySelector('svg'))
              .select('#bars')
              .selectAll('line')
              .nodes()
              .length,
          (barChart.slices.length + 1) * barChart.metrics.length);
      done();
    };
    setTimeout(fillData, 0);
  });

  test('BarCheckForBoundedValueEvalComparison', done => {
    barChart = fixture('main');

    const fillData = () => {
      barChart.data = BOUNDED_VALUE_DATA;
      barChart.dataCompare = BOUNDED_VALUE_DATA;
      barChart.evalName = 'EvalUno';
      barChart.evalNameCompare = 'EvalDos';
      barChart.metrics = [
        'post_export_metrics/false_negative_rate@0.30',
        'post_export_metrics/false_negative_rate@0.50'
      ];
      barChart.baseline = SLICES[0];
      barChart.slices = SLICES.slice(1);
      setTimeout(checkValue, 100);
    };

    const checkValue = () => {
      // One group for every eval-slice pair
      const numClusters = 2 * (barChart.slices.length + 1);
      assert.equal(
          d3.select(barChart.shadowRoot.querySelector('svg'))
              .select('#bars')
              .selectAll('g')
              .nodes()
              .length,
          numClusters);

      // One bar for every cluster-metric pair
      const numBars = numClusters * barChart.metrics.length;
      assert.equal(
          d3.select(barChart.shadowRoot.querySelector('svg'))
              .select('#bars')
              .selectAll('rect')
              .nodes()
              .length,
          numBars);

      // One line per bar
      assert.equal(
          d3.select(barChart.shadowRoot.querySelector('svg'))
              .select('#bars')
              .selectAll('line')
              .nodes()
              .length,
          numBars);
      done();
    };
    setTimeout(fillData, 0);
  });

  test('BarCheckForBoundedValueEvalComparisonWithUnequalSlices', done => {
    barChart = fixture('main');

    const fillData = () => {
      barChart.data = BOUNDED_VALUE_DATA;
      // Remove one slice to make the difference between data and dataCompare.
      barChart.dataCompare =
          BOUNDED_VALUE_DATA.slice(0, BOUNDED_VALUE_DATA.length - 1);
      barChart.evalName = 'Eval A';
      barChart.evalNameCompare = 'Eval B';
      barChart.metrics = [
        'post_export_metrics/false_negative_rate@0.30',
      ];
      barChart.baseline = SLICES[0];
      barChart.slices = SLICES.slice();
      setTimeout(checkValue, 100);
    };

    const checkValue = () => {
      // One group for every eval-slice pair
      const numClusters = 2 * (barChart.slices.length + 1);
      const numBars = numClusters * barChart.metrics.length;
      let bars = d3.select(barChart.shadowRoot.querySelector('svg'))
                     .select('#bars')
                     .selectAll('rect')
                     .nodes();
      assert.equal(bars.length, numBars);
      // The last bar's height is 0.
      assert.equal(bars[bars.length - 1].getAttribute('height'), 0);
      done();
    };
    setTimeout(fillData, 0);
  });

  const d3DataObject = (fullSliceName, evalName) => {
    const obj = new Object();
    obj.fullSliceName = fullSliceName;
    obj.evalName = evalName;
    return obj;
  };

  const D3_DATA = [
    d3DataObject('Overall', 'eval1'), d3DataObject('slice:A', 'eval1'),
    d3DataObject('slice:B', 'eval1'), d3DataObject('Overall', 'eval2'),
    d3DataObject('slice:A', 'eval2'), d3DataObject('slice:B', 'eval2')
  ];

  const ALTERNATING_EVALS =
      ['eval1', 'eval2', 'eval1', 'eval2', 'eval1', 'eval2'];
  const ALTERNATING_SLICES =
      ['Overall', 'slice:A', 'slice:B', 'Overall', 'slice:A', 'slice:B'];

  const SLICE_SORT = 'Slice';
  const EVAL_SORT = 'Eval';

  test('SortBySlice_BaselineOverall', done => {
    barChart = fixture('main');

    // We define these so that barChart knows we are comparing evals
    barChart.data = BOUNDED_VALUE_DATA;
    barChart.dataCompare = BOUNDED_VALUE_DATA;

    const sortedD3data = D3_DATA.sort(barChart.sortD3_('Overall', SLICE_SORT));
    const actualSlices = sortedD3data.map(d => d.fullSliceName);
    const actualEvals = sortedD3data.map(d => d.evalName);

    const expectedSlicesOverall =
        ['Overall', 'Overall', 'slice:A', 'slice:A', 'slice:B', 'slice:B'];

    assert.sameOrderedMembers(actualSlices, expectedSlicesOverall);
    assert.sameOrderedMembers(actualEvals, ALTERNATING_EVALS);
  });

  test('SortBySlice_BaselineA', done => {
    barChart = fixture('main');

    // We define these so that barChart knows we are comparing evals
    barChart.data = BOUNDED_VALUE_DATA;
    barChart.dataCompare = BOUNDED_VALUE_DATA;

    const sortedD3data = D3_DATA.sort(barChart.sortD3_('slice:A', SLICE_SORT));
    const actualSlices = sortedD3data.map(d => d.fullSliceName);
    const actualEvals = sortedD3data.map(d => d.evalName);

    const expectedSlicesA = [
      'slice:A',
      'slice:A',
      'Overall',
      'Overall',
      'slice:B',
      'slice:B',
    ];

    assert.sameOrderedMembers(actualSlices, expectedSlicesA);
    assert.sameOrderedMembers(actualEvals, ALTERNATING_EVALS);
  });

  test('SortBySlice_BaselineB', done => {
    barChart = fixture('main');

    // We define these so that barChart knows we are comparing evals
    barChart.data = BOUNDED_VALUE_DATA;
    barChart.dataCompare = BOUNDED_VALUE_DATA;

    const sortedD3data = D3_DATA.sort(barChart.sortD3_('slice:B', SLICE_SORT));
    const actualSlices = sortedD3data.map(d => d.fullSliceName);
    const actualEvals = sortedD3data.map(d => d.evalName);

    const expectedSlicesB = [
      'slice:B',
      'slice:B',
      'Overall',
      'Overall',
      'slice:A',
      'slice:A',
    ];

    assert.sameOrderedMembers(actualSlices, expectedSlicesB);
    assert.sameOrderedMembers(actualEvals, ALTERNATING_EVALS);
  });

  test('SortByEval', done => {
    barChart = fixture('main');

    // We define these so that barChart knows we are comparing evals
    barChart.data = BOUNDED_VALUE_DATA;
    barChart.dataCompare = BOUNDED_VALUE_DATA;

    const sortedD3data = D3_DATA.sort(barChart.sortD3_('Overall', EVAL_SORT));
    const actualSlices = sortedD3data.map(d => d.fullSliceName);
    const actualEvals = sortedD3data.map(d => d.evalName);

    const expectedEvals =
        ['eval1', 'eval1', 'eval1', 'eval2', 'eval2', 'eval2'];

    assert.sameOrderedMembers(actualSlices, ALTERNATING_SLICES);
    assert.sameOrderedMembers(actualEvals, expectedEvals);
  });
});
