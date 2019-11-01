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
  const TABLE_DATA = [
    {
      'slice': 'col:1',
      'metrics': {
        'logisticLoss': 0.7,
        'averageLabel': 0.5,
        'count': '1000000',
        'auprc': 0.7,
        'boundedAuc': {
          'value': 0.611111,
          'lowerBound': 0.6011111,
          'upperBound': 0.6211111
        },
      }
    },
    {
      'slice': 'col:2',
      'metrics': {
        'logisticLoss': 0.72,
        'averageLabel': 0.52,
        'count': '1000002',
        'auprc': 0.72,
        'boundedAuc':
            {'value': 0.612, 'lowerBound': 0.602, 'upperBound': 0.622},
      }
    },
    {
      'slice': 'col:3',
      'metrics': {
        'logisticLoss': 0.73,
        'count': '2000003',
        'auprc': 0.73,
        'boundedAuc':
            {'value': 0.613, 'lowerBound': 0.603, 'upperBound': 0.623},
      },
    }
  ];

  const METRICS = ['logisticLoss', 'count', 'boundedAuc', 'auprc'];

  const HEADER_OVERRIDE = {'logisticLoss': 'loss'};

  const EXAMPLE_COUNTS = [34, 84, 49];

  let table;

  test('ComputingTableData', done => {
    table = fixture('test-fixture');

    const fillData = () => {
      table.metrics = METRICS;
      table.data = TABLE_DATA;
      table.headerOverride = HEADER_OVERRIDE;
      table.exampleCounts = EXAMPLE_COUNTS;
      setTimeout(CheckProperties, 500);
    };

    const CheckProperties = () => {
      const expected_data = [
        ['feature', 'loss', 'count', 'boundedAuc', 'auprc'],
        ['col:1', '0.7', '1000000', '0.61111 (0.60111, 0.62111)', '0.7'],
        ['col:2', '0.72', '1000002', '0.61200 (0.60200, 0.62200)', '0.72'],
        ['col:3', '0.73', '2000003', '0.61300 (0.60300, 0.62300)', '0.73'],
      ];

      assert.equal(table.plotData_.length, expected_data.length);
      for (var i = 0; i < 4; i++) {
        for (var j = 0; j < 5; j++) {
          assert.equal(table.plotData_[i][j], expected_data[i][j]);
        }
      }

      table.shadowRoot.querySelectorAll('.table-row').forEach(function (row) {
        const cells = row.querySelectorAll('.table-entry');
        for (var i = 0; i < cells.length; i++) {
          const content = cells[i].textContent.trim();
          if (i % 2) {
            assert.isTrue(content[content.length-1] === '%');
          } else {
            assert.isTrue(content[content.length-1] != '%');
          }
        }
      });

      done();
    };

    setTimeout(fillData, 0);
  });
});
