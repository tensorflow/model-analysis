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
  const NUMBER_OF_BUCKETS = 4;
  const DEFAULT_BUCKETS = [
    {
      'lowerThresholdInclusive': -Infinity,
      'upperThresholdExclusive': 0,
      'numWeightedExamples': 0,
      'totalWeightedLabel': 0,
      'totalWeightedRefinedPrediction': 0,
    },
    {
      'lowerThresholdInclusive': 0,
      'upperThresholdExclusive': 0.125,
      'numWeightedExamples': 1,
      'totalWeightedLabel': 0.5,
      'totalWeightedRefinedPrediction': 0.5,
    },
    {
      'lowerThresholdInclusive': 0.125,
      'upperThresholdExclusive': 0.25,
      'numWeightedExamples': 10,
      'totalWeightedLabel': 7.5,
      'totalWeightedRefinedPrediction': 0.5,
    },
    {
      'lowerThresholdInclusive': 0.25,
      'upperThresholdExclusive': 0.375,
      'numWeightedExamples': 100,
      'totalWeightedLabel': 90,
      'totalWeightedRefinedPrediction': 81,
    },
    {
      'lowerThresholdInclusive': 0.375,
      'upperThresholdExclusive': 0.5,
      'numWeightedExamples': 60,
      'totalWeightedLabel': 20,
      'totalWeightedRefinedPrediction': 20,
    },
    {
      'lowerThresholdInclusive': 0.5,
      'upperThresholdExclusive': 0.625,
      'numWeightedExamples': 100,
      'totalWeightedLabel': 16,
      'totalWeightedRefinedPrediction': 64,
    },
    {
      'lowerThresholdInclusive': 0.625,
      'upperThresholdExclusive': 0.75,
      'numWeightedExamples': 50,
      'totalWeightedLabel': 20,
      'totalWeightedRefinedPrediction': 30,
    },
    {
      'lowerThresholdInclusive': 0.75,
      'upperThresholdExclusive': 0.875,
      'numWeightedExamples': 30,
      'totalWeightedLabel': 11,
      'totalWeightedRefinedPrediction': 21,
    },
    {
      'lowerThresholdInclusive': 0.875,
      'upperThresholdExclusive': 1.0,
      'numWeightedExamples': 70,
      'totalWeightedLabel': 39,
      'totalWeightedRefinedPrediction': 54,
    },
    {
      'lowerThresholdInclusive': 1,
      'upperThresholdExclusive': Infinity,
      'numWeightedExamples': 0,
      'totalWeightedLabel': 0,
      'totalWeightedRefinedPrediction': 0,
    },
  ];

  let plot;

  /**
   * @param {{fit: (string|undefined),
   *     color: (string|undefined),
   *     size: (string|undefined),
   *     scale: (string|undefined)
   * }} config
   */
  function setUpFixture({fit, color, size, scale}) {
    plot = fixture('test-plot');
    if (fit) {
      plot.fit = fit;
    }
    if (color) {
      plot.color = color;
    }
    if (size) {
      plot.size = size;
    }
    if (scale) {
      plot.scale = scale;
    }

    plot.buckets = DEFAULT_BUCKETS;
    plot.numberOfBuckets = NUMBER_OF_BUCKETS;
  }

  test('SetColorOverrides', () => {
    setUpFixture({});
    const highValue = 'rgb(123, 123, 123)';
    const lowValue = 'teal';
    const maxValue = 654;
    const minValue = 321;

    plot.overrides = {
      'colorHighValue': highValue,
      'colorLowValue': lowValue,
      'colorMaxValue': maxValue,
      'colorMinValue': minValue
    };

    var colorOptions = plot.$['plot']['options']['colorAxis'];
    assert.equal(colorOptions['colors'][0], lowValue);
    assert.equal(colorOptions['colors'][1], highValue);
    assert.equal(colorOptions['maxValue'], maxValue);
    assert.equal(colorOptions['minValue'], minValue);
  });

  test('SetSizeOverrides', () => {
    setUpFixture({});
    const maxRadius = 456;
    const minRadius = 123;
    const maxValue = 654;
    const minValue = 321;

    plot.overrides = {
      'sizeMaxRadius': maxRadius,
      'sizeMinRadius': minRadius,
      'sizeMaxValue': maxValue,
      'sizeMinValue': minValue
    };

    const sizeOptions = plot.$['plot']['options']['sizeAxis'];
    assert.equal(sizeOptions['maxSize'], maxRadius);
    assert.equal(sizeOptions['minSize'], minRadius);
    assert.equal(sizeOptions['maxValue'], maxValue);
    assert.equal(sizeOptions['minValue'], minValue);
  });

  test('SetCheckViewWindowOverrides', () => {
    setUpFixture({});

    const hAxis = plot.$['plot']['options']['hAxis'];
    assert.equal(hAxis['minValue'], 0);
    assert.equal(hAxis['maxValue'], 1);
    assert.equal(hAxis['viewWindow']['min'], 0);
    assert.equal(hAxis['viewWindow']['max'], 1);

    const vAxis = plot.$['plot']['options']['vAxis'];
    assert.equal(vAxis['minValue'], 0);
    assert.equal(vAxis['maxValue'], 1);
    assert.equal(vAxis['viewWindow']['min'], 0);
    assert.equal(vAxis['viewWindow']['max'], 1);
  });

  test('CreateChartDataWithoutRebucketing', () => {
    setUpFixture({});
    plot.numberOfBuckets = 0;

    const plotData = plot.plotData_;
    assert.equal(plotData.length, 9);
    assert.deepEqual(
        plotData[0],
        ['bucket', 'prediction', 'label', 'color: error', 'size: log(weight)']);
    assert.equal(plotData[1][0], '');
    assert.equal(plotData[1][1], 0.5);
    assert.equal(plotData[1][2], 0.5);
    assert.equal(plotData[2][3], 0.7);
    assert.equal(plotData[2][4], 1);
  });

  test('DefaultsToLinearScale', () => {
    setUpFixture({});

    assert.isUndefined(plot.options_['hAxis']['logScale']);
    assert.isUndefined(plot.options_['vAxis']['logScale']);
  });

  test('OverrideToLogScale', () => {
    setUpFixture({scale: tfma.PlotScale.LOG});

    assert.isTrue(plot.options_['hAxis']['logScale']);
    assert.isTrue(plot.options_['vAxis']['logScale']);
  });

  test('CreateChartDataWithRebucketing', () => {
    setUpFixture({});

    const plotData = plot.plotData_;
    assert.equal(plotData.length, 5);

    // The last two buckets from DEFAULT_BUCKETS should be merged.
    assert.equal(plotData[4][0], '');
    assert.equal(plotData[4][1], 0.75);
    assert.equal(plotData[4][2], 0.5);
    assert.equal(plotData[4][3], 0.25);
    assert.equal(plotData[4][4], 2);
  });

  test('CreateChartDataWithLeastSquareFit', () => {
    setUpFixture({
      fit: tfma.PlotFit.LEAST_SQUARE,
    });
    plot.numberOfBuckets = 0;

    // Set up the data so that data fits a line perfectly.
    const SLOPE = 0.5;
    const INTERCEPT = 0.125;
    plot.buckets = [
      {
        'lowerThresholdInclusive': 0,
        'upperThresholdExclusive': 0.25,
        'numWeightedExamples': 1,
        'totalWeightedLabel': 0.25 * SLOPE + INTERCEPT,
        'totalWeightedRefinedPrediction': 0.25,
      },
      {
        'lowerThresholdInclusive': 0.25,
        'upperThresholdExclusive': 0.5,
        'numWeightedExamples': 1,
        'totalWeightedLabel': 0.5 * SLOPE + INTERCEPT,
        'totalWeightedRefinedPrediction': 0.5,
      },
      {
        'lowerThresholdInclusive': 0.5,
        'upperThresholdExclusive': 0.75,
        'numWeightedExamples': 1,
        'totalWeightedLabel': 0.75 * SLOPE + INTERCEPT,
        'totalWeightedRefinedPrediction': 0.75,
      },
      {
        'lowerThresholdInclusive': 0.75,
        'upperThresholdExclusive': 1,
        'numWeightedExamples': 1,
        'totalWeightedLabel': 1 * SLOPE + INTERCEPT,
        'totalWeightedRefinedPrediction': 1,
      }
    ];

    const plotData = plot.plotData_;

    assert.equal(plotData[1][3], 0);
    assert.equal(plotData[2][3], 0);
    assert.equal(plotData[3][3], 0);
    assert.equal(plotData[4][3], 0);
  });

  test('CreateChartDataWithColorAndSizeOverride', () => {
    setUpFixture({
      color: tfma.PlotHighlight.WEIGHTS,
      size: tfma.PlotHighlight.ERROR,
    });

    const plotData = plot.plotData_;
    assert.equal(plotData.length, 5);
    assert.deepEqual(
        plotData[0],
        ['bucket', 'prediction', 'label', 'color: log(weight)', 'size: error']);
    assert.equal(plotData[4][0], '');
    assert.equal(plotData[4][1], 0.75);
    assert.equal(plotData[4][2], 0.5);
    assert.equal(plotData[4][3], 2);
    assert.equal(plotData[4][4], 0.25);
  });
});
