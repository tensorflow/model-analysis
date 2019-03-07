/**
 * Copyright 2018 Google LLC
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
suite('tests', () => {
  const DEFAULT_DATA = {
    'A': {
      'A': {
        'value': 100,
        'tooltip': 'Click',
        'details': '(row A, col A)',
      },
      'B': {
        'value': 50,
        'tooltip': 'Click',
        'details': '(row A, col B)',
      },
    },
    'B': {
      'A': {
        'value': 10,
        'tooltip': 'Click',
        'details': '(row B, col A)',
      },
      'B': {
        'value': 30,
        'tooltip': 'Click',
        'details': '(row B, col B)',
      },
    }
  };

  /**
   * Test component element.
   * @type {!Element}
   */
  let element;

  const next = (cb) => {
    setTimeout(cb, 1);
  };

  const run = (cb, data) => {
    element = fixture('matrix');
    element.expanded = true;
    element.minValue = 10;
    element.maxValue = 100;
    element.rowLabel = 'Row';
    element.columnLabel = 'Column';
    element.data = data;
    next(cb);
  };

  test('parseData', done => {
    const checkMatrixContent = () => {
      const headerRow =
          element.shadowRoot.querySelector('.matrix-row:nth-child(2)');
      assert.deepEqual(
          headerRow.textContent.trim().split(/[\s\n]+/), ['A', 'B']);

      const contentRow1 =
          element.shadowRoot.querySelector('.matrix-row:nth-child(3)');
      assert.deepEqual(
          contentRow1.textContent.trim().split(/[\s\n]+/), ['A', '100', '50']);

      const contentRow2 =
          element.shadowRoot.querySelector('.matrix-row:nth-child(4)');
      assert.deepEqual(
          contentRow2.textContent.trim().split(/[\s\n]+/), ['B', '10', '30']);

      done();
    };

    run(checkMatrixContent, DEFAULT_DATA);
  });

  test('sortByColumn', done => {
    const sortByColumnA = () => {
      const headerRow =
          element.shadowRoot.querySelector('.matrix-row:nth-child(2)');
      const headerA = headerRow.querySelector('.header:nth-child(1)');
      headerA.click();
      next(checkMatrixContent);
    };

    const checkMatrixContent = () => {
      const headerRow =
          element.shadowRoot.querySelector('.matrix-row:nth-child(2)');
      assert.deepEqual(
          headerRow.textContent.trim().split(/[\s\n]+/), ['A', 'B']);

      const contentRow1 =
          element.shadowRoot.querySelector('.matrix-row:nth-child(3)');
      assert.deepEqual(
          contentRow1.textContent.trim().split(/[\s\n]+/), ['B', '10', '30']);

      const contentRow2 =
          element.shadowRoot.querySelector('.matrix-row:nth-child(4)');
      assert.deepEqual(
          contentRow2.textContent.trim().split(/[\s\n]+/), ['A', '100', '50']);

      done();
    };

    run(sortByColumnA, DEFAULT_DATA);
  });

  test('sortByRow', done => {
    const sortByRowA = () => {
      const contentRow1 =
          element.shadowRoot.querySelector('.matrix-row:nth-child(3)');
      const headerB = contentRow1.querySelector('.header');
      headerB.click();
      next(checkMatrixContent);
    };

    const checkMatrixContent = () => {
      const headerRow =
          element.shadowRoot.querySelector('.matrix-row:nth-child(2)');
      assert.deepEqual(
          headerRow.textContent.trim().split(/[\s\n]+/), ['B', 'A']);

      const contentRow1 =
          element.shadowRoot.querySelector('.matrix-row:nth-child(3)');
      assert.deepEqual(
          contentRow1.textContent.trim().split(/[\s\n]+/), ['A', '50', '100']);

      const contentRow2 =
          element.shadowRoot.querySelector('.matrix-row:nth-child(4)');
      assert.deepEqual(
          contentRow2.textContent.trim().split(/[\s\n]+/), ['B', '30', '10']);

      done();
    };

    run(sortByRowA, DEFAULT_DATA);
  });

  test('showCellDetail', done => {
    const clickCellAA = () => {
      const cell =
          element.shadowRoot.querySelector('.matrix-row:nth-child(3) .cell');
      cell.click();
      next(checkDetailsAA);
    };

    const checkDetailsAA = () => {
      assert.equal(
          element.$['details'].textContent.trim().split(/[\s\n]+/).join(''),
          '(rowA,colA)');
      next(clickCellBA);
    };

    const clickCellBA = () => {
      const cell =
          element.shadowRoot.querySelector('.matrix-row:nth-child(4) .cell');
      cell.click();
      next(checkDetailsBA);
    };

    const checkDetailsBA = () => {
      assert.equal(
          element.$['details'].textContent.trim().split(/[\s\n]+/).join(''),
          '(rowB,colA)');

      done();
    };

    run(clickCellAA, DEFAULT_DATA);
  });

  test('omitVisualizationIfTooManyClasses', done => {
    const makeData = (count) => {
      const data = {};
      for (let i = 0; i < count; i++) {
        const row = {};
        for (let j = 0; j < count; j++) {
          row['C' + j] = {
            'value': i + j,
            'tooltip': 'omit',
            'details': 'omit',
          };
        }
        data['R' + i] = row;
      }
      return data;
    };

    const checkText = () => {
      assert.equal(element.$.root.textContent.trim(), 'Omitted');
      done();
    };

    run(checkText, makeData(128));
  });

  test('pivot', done => {
    const configure = () => {
      element.minValue = -10;
      element.maxValue = 30;
      element.pivot = 10;
      next(checkScale);
    };

    const checkScale = () => {
      const cells = element.$.root.querySelectorAll('div.cell');
      assert.equal(cells.length, 4);
      assert.isTrue(cells[0].classList.contains('b12'));
      assert.isTrue(cells[1].classList.contains('b0'));
      assert.isTrue(cells[2].classList.contains('b10'));
      assert.isTrue(cells[3].classList.contains('b2'));

      done();
    };

    run(configure, {
      'A': {
        'A': {
          'value': -5,
          'tooltip': 'Click',
          'details': '(row A, col A)',
        },
        'B': {
          'value': 10,
          'tooltip': 'Click',
          'details': '(row A, col A)',
        },
      },
      'B': {
        'A': {
          'value': 23,
          'tooltip': 'Click',
          'details': '(row A, col A)',
        },
        'B': {
          'value': 8,
          'tooltip': 'Click',
          'details': '(row A, col A)',
        },
      },
    });
  });
});
