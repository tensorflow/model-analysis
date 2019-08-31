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
goog.module('tfma.tests.UtilTest');

goog.setTestOnly();

const Util = goog.require('tfma.Util');
const testSuite = goog.require('goog.testing.testSuite');
goog.require('goog.testing.jsunit');


const output1 = 'output1';
const output2 = 'output2';
const class1 = 'classId:1';
const class2 = 'classId:2';
const top3 = 'topK:3';
const metric1 = {
  some: 1,
  value: 2
};
const metric2 = {
  more: 2,
  value: 3
};
const metric3 = {
  other: 3,
  value: 4
};
const metric4 = {
  stillMore: 4,
  value: 3
};
const metric5 = {
  evenMore: 3,
  value: 2
};

const TEST_DATA = {
  [output1]: {
    [class1]: {
      auc: 0.81,
      complex: metric1,
    },
  },
  [output2]: {
    [class2]: {
      auc: 0.82,
      complex: metric2,
    },
    [top3]: {
      auc: 0.83,
      complex: metric3,
    },
    '': {
      auc: 0.80,
      complex: metric4,
    }
  },
  '': {
    '': {
      auc: 0.84,
      complex: metric5,
    },
  },
};

testSuite({
  testCreateConfigsList: () => {
    assertArrayEquals(
        [
          {outputName: output1, classId: class1},
          {outputName: output2, classId: class2},
          {outputName: output2, classId: top3},
        ],
        Util.createConfigsList(
            {[output1]: [class1], [output2]: [class2, top3]}));
  },

  testMergeMetricsForSelectedConfigsListNoPrefixIfOnlyOneConfigSelected: () => {
    const selectedConfigs = [{outputName: output1, classId: class1}];

    assertObjectEquals(
        {auc: 0.81, complex: metric1},
        Util.mergeMetricsForSelectedConfigsList(TEST_DATA, selectedConfigs));
  },

  testMergeMetricsForSelectedConfigsListAddPrefixIfMoreThanOneSelected: () => {
    const selectedConfigs = [
      {outputName: output1, classId: class1},
      {outputName: output2, classId: top3}
    ];
    const class1Auc = output1 + '/' + class1 + '/' +
        'auc';
    const class1Complex = output1 + '/' + class1 + '/' +
        'complex';
    const top3Auc = output2 + '/' + top3 + '/' +
        'auc';
    const top3Complex = output2 + '/' + top3 + '/' +
        'complex';
    assertObjectEquals(
        {
          [class1Auc]: 0.81,
          [class1Complex]: metric1,
          [top3Auc]: 0.83,
          [top3Complex]: metric3,
        },
        Util.mergeMetricsForSelectedConfigsList(TEST_DATA, selectedConfigs));
  },

  testMergeMetricsForSelectedConfigsListAddCompactPrefixIfSimilarSelected:
      () => {
        const selectedConfigs = [
          {outputName: output1, classId: class1},
          {outputName: output2, classId: class2}
        ];
        const class1Auc = output1 + '/1/auc';
        const class1Complex = output1 + '/1/complex';
        const class2Auc = output2 + '/2/auc';
        const class2Complex = output2 + '/2/complex';
        assertObjectEquals(
            {
              [class1Auc]: 0.81,
              [class1Complex]: metric1,
              [class2Auc]: 0.82,
              [class2Complex]: metric2,
            },
            Util.mergeMetricsForSelectedConfigsList(
                TEST_DATA, selectedConfigs));
      },

  testMergeMetricsForSelectedConfigsListAddCompactPrefixIfNoClassId: () => {
    const selectedConfigs = [
      {outputName: output1, classId: class1}, {outputName: output2, classId: ''}
    ];
    const class1Auc = output1 + '/1/auc';
    const class1Complex = output1 + '/1/complex';
    const class2Auc = output2 + '/auc';
    const class2Complex = output2 + '/complex';
    assertObjectEquals(
        {
          [class1Auc]: 0.81,
          [class1Complex]: metric1,
          [class2Auc]: 0.80,
          [class2Complex]: metric4,
        },
        Util.mergeMetricsForSelectedConfigsList(TEST_DATA, selectedConfigs));
  },

  testMergeMetricsForSelectedConfigsListSkipsBlacklistedMetrics: () => {
    const selectedConfigs = [
      {outputName: output1, classId: class1},
      {outputName: output2, classId: top3}
    ];
    const class1Complex = output1 + '/' + class1 + '/' +
        'complex';
    const top3Complex = output2 + '/' + top3 + '/' +
        'complex';

    assertObjectEquals(
        {
          [class1Complex]: metric1,
          [top3Complex]: metric3,
        },
        Util.mergeMetricsForSelectedConfigsList(
            TEST_DATA, selectedConfigs, {auc: 1}));
  },

  testMergeMetricsForSelectedConfigsListSkipsOutputIfEmpty: () => {
    const selectedConfigs = [
      {outputName: '', classId: ''},
      {outputName: output1, classId: class1},
    ];

    const merged =
        Util.mergeMetricsForSelectedConfigsList(TEST_DATA, selectedConfigs);
    assertEquals(metric5, merged['complex']);
  },

  testMergeMetricsForSelectedConfigsListSkipsClassIdIfEmpty: () => {
    const selectedConfigs = [
      {outputName: output1, classId: class1},
      {outputName: output2, classId: ''},
    ];

    const merged =
        Util.mergeMetricsForSelectedConfigsList(TEST_DATA, selectedConfigs);
    assertEquals(metric4, merged[output2 + '/complex']);
  },
});
