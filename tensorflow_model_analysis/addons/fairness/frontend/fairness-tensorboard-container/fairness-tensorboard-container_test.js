/**
 * Copyright 2021 Google LLC
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

suite('fairness-tensorboard-container tests', () => {
  const SLICES_NAMES = [
    'Overall', 'Slice:1', 'Slice:2', 'Slice:3', 'Slice:4', 'Slice:5', 'Slice:6',
    'Slice:7', 'Slice:8', 'Slice:9', 'Slice:10', 'Slice:11', 'Slice:12',
    'Slice:13', 'Slice:14', 'Slice:15', 'Slice:16'
  ];

  let fairnessContainer;
  const generateSlicingMetrics = () => {
    return SLICES_NAMES.map((slice) => {
      return {
        'slice': slice,
        'sliceValue': slice.split(':')[1] || 'Overall',
        'metrics': {
          'flip_rate/overall@0.2': {
            'doubleValue': Math.random(),
          },
          'lift@2': {
            'doubleValue': Math.random(),
          },
          'lift@23': {
            'doubleValue': Math.random(),
          },
          'accuracy': {
            'doubleValue': Math.random(),
          },
          'post_export_metrics/positive_rate@0.50': {
            'doubleValue': Math.random(),
          },
          'post_export_metrics/true_positive_rate@0.50': {
            'doubleValue': Math.random(),
          },
          'post_export_metrics/false_positive_rate@0.50': {
            'doubleValue': NaN,
          },
          'post_export_metrics/positive_rate@0.60': {
            'doubleValue': Math.random(),
          },
          'post_export_metrics/true_positive_rate@0.60': {
            'doubleValue': Math.random(),
          },
          'post_export_metrics/false_positive_rate@0.60': {
            'doubleValue': Math.random(),
          },
          'totalWeightedExamples': {'doubleValue': 2000 * (Math.random() + 0.8)}
        }
      };
    });
  };

  setup(() => {
    fairnessContainer = fixture('test-fixture');
  });

  test('testRunSelectorHidden', done => {
    const fillData = () => {
      fairnessContainer.slicingMetrics_ = generateSlicingMetrics();
      setTimeout(checkRunSelectorIsInvisible, 0);
    };
    const checkRunSelectorIsInvisible = () => {
      let runSelector =
          fairnessContainer.shadowRoot.querySelector('#run-selector');
      assert.equal(runSelector.hidden, true);
      done();
    };
    setTimeout(fillData, 0);
  });

  test('testRunSelectorVisibleWhenEvalRunsAvailable', done => {
    const fillData = () => {
      fairnessContainer.slicingMetrics_ = generateSlicingMetrics();
      fairnessContainer.evaluationRuns_ = ['1', '2', '3'];
      setTimeout(checkRunSelectorIsVisible, 0);
    };
    const checkRunSelectorIsVisible = () => {
      let runSelector =
          fairnessContainer.shadowRoot.querySelector('#run-selector');
      assert.equal(runSelector.hidden, false);
      done();
    };
    setTimeout(fillData, 0);
  });

  test('testModelComparisonFlow', done => {
    const checkCheckbox = () => {
      let checkbox =
          fairnessContainer.shadowRoot.querySelector('#model-comparison');
      assert.equal(checkbox.checked, false);

      let runSelector =
          fairnessContainer.shadowRoot.querySelector('#div-to-compare');
      assert.equal(runSelector.hidden, true);

      checkbox.checked = true;
      assert.equal(runSelector.hidden, false);
      done();
    };

    setTimeout(checkCheckbox, 0);
  });
});
