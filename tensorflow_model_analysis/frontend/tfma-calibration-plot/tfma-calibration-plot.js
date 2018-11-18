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
Polymer({
  is: 'tfma-calibration-plot',

  properties: {
    /**
     * An array of buckets in json.
     * @type {!Array<!Object>}
     */
    buckets: {type: Array},

    /**
     * The size of each bucket.
     * @type {number}
     */
    bucketSize: {type: Number},

    /**
     * How to determine the error in each prediction / label pair. For available
     * values, see enum tfma.PlotFit. Defaults to perfectly calibrated (y = x)
     * which is suitable for most regression problems. For ranking problems,
     * since the interesting part is that label / prediction pair increase
     * monotonically, sometimes a least square fit would be better at
     * highlighting the outliers.
     * @type {string}
     */
    fit: {type: String, value: tfma.PlotFit.PERFECT},

    /**
     * What does the color of each dot highlight. For available values, see enum
     * tfma.PlotHighlight. Defaults to use error.
     * @type {string}
     */
    color: {type: String, value: tfma.PlotHighlight.ERROR},

    /**
     * What does the size of each dot highlight. For available values, see enum
     * tfma.PlotHighlight. Default to use weight.
     * @type {string}
     */
    size: {type: String, value: tfma.PlotHighlight.WEIGHTS},

    /**
     * What is the scale for the x and y axes. For available values, see enum
     * tfma.PlotScale.
     * @type {string}
     */
    scale: {type: String, value: tfma.PlotScale.LINEAR},

    /**
     * Overrides that should be applied to the chart.
     * @type {{
     *   colorHighValue: (number|undefined),
     *   colorLowValue: (number|undefined),
     *   colorMaxValue: (number|undefined),
     *   colorMinValue: (number|undefined),
     *   sizeMaxRadius: (number|undefined),
     *   sizeMinRadius: (number|undefined),
     *   sizeMaxValue: (number|undefined),
     *   sizeMinValue: (number|undefined),
     *   title: (string|undefined),
     * }}
     */
    overrides: {type: Object, value: {}},

    /**
     * Options for the bubble chart.
     * @type {!Object}
     */
    options: {
      type: Object,
      computed: 'computeOptions_(color, size, scale, overrides)'
    },

    /**
     * The header to use for the plot.
     * @private {!Array<string>}
     */
    header_: {type: Array, computed: 'getHeader_(scale, color, size)'},

    /**
     * The data to be plotted in the bubble chart.
     * @private {!Array<!Array<string|number>>}
     */
    plotData_: {
      type: Array,
      computed: 'computePlotData_(buckets, header_, fit, scale, color, size, ' +
          'bucketSize)'
    },
  },

  /**
   * @param {string} color
   * @param {string} size
   * @param {string} scale
   * @param {!Object} overrides
   * @return {!Object} The options object used for configuring the google-chart
   *     object.
   * @private
   */
  computeOptions_: function(color, size, scale, overrides) {
    var options = {
      'title': 'Calibration Plot',
      'hAxis': {
        'title': 'Average Prediction',
        'minValue': 0,
        'maxValue': 1,
        // Forces displayed value range to [0, 1].
        'viewWindow': {'min': 0, 'max': 1}
      },
      'vAxis': {'title': 'Average Label', 'minValue': 0, 'maxValue': 1,
        // Forces displayed value range [0, 1].
        'viewWindow': {'min': 0, 'max': 1}
      },
      'bubble': {'textStyle': {'fontSize': 11}},
      'colorAxis':
          {'colors': ['#F0F0F0', '#0A47A4'], 'minValue': 0, 'maxValue': 10},
      'sizeAxis': {'minValue': 0, 'maxValue': 0.5, 'minSize': 2, 'maxSize': 12},
      'explorer':
          {'actions': ['dragToPan', 'scrollToZoom', 'rightClickToReset']},
    };

    if (scale == tfma.PlotScale.LOG) {
      options['hAxis']['logScale'] = true;
      options['vAxis']['logScale'] = true;
    }

    // Handle color.
    var colorAxis = options['colorAxis'];
    if (color == tfma.PlotHighlight.ERROR) {
      colorAxis['maxValue'] = 0.5;
    }

    // Apply color overrides.
    if (overrides['colorLowValue']) {
      colorAxis['colors'][0] = overrides['colorLowValue'];
    }
    if (overrides['colorHighValue']) {
      colorAxis['colors'][1] = overrides['colorHighValue'];
    }
    if (overrides['colorMinValue'] >= 0) {
      colorAxis['minValue'] = overrides['colorMinValue'];
    }
    if (overrides['colorMaxValue']) {
      colorAxis['maxValue'] = overrides['colorMaxValue'];
    }

    // Handle size.
    var sizeAxis = options['sizeAxis'];
    if (size == tfma.PlotHighlight.WEIGHTS) {
      sizeAxis['maxValue'] = 10;
    }

    // Apply size overrides.
    if (overrides['sizeMinRadius']) {
      sizeAxis['minSize'] = overrides['sizeMinRadius'];
    }
    if (overrides['sizeMaxRadius']) {
      sizeAxis['maxSize'] = overrides['sizeMaxRadius'];
    }
    if (overrides['sizeMinValue'] >= 0) {
      sizeAxis['minValue'] = overrides['sizeMinValue'];
    }
    if (overrides['sizeMaxValue']) {
      sizeAxis['maxValue'] = overrides['sizeMaxValue'];
    }

    if (overrides['title']) {
      options['title'] = overrides['title'];
    }

    return options;
  },

  /**
   * @param {string} scale
   * @param {string} color
   * @param {string} size
   * @return {!Array<string>} An array of strings that will be used as the title
   *     of the calibration plot.
   * @private
   */
  getHeader_: function(scale, color, size) {
    var header = ['bucket', 'prediction', 'label', 'color: ', 'size: '];

    if (scale == tfma.PlotScale.LOG) {
      header[1] = 'log(prediction)';
      header[2] = 'log(label)';
    }

    header[3] += color == tfma.PlotHighlight.ERROR ? 'error' : 'log(weight)';
    header[4] += size == tfma.PlotHighlight.WEIGHTS ? 'log(weight)' : 'error';

    return header;
  },

  /**
   * @param {!Array<!Object>} buckets
   * @param {!Array<string>} header
   * @param {string} fit
   * @param {string} scale
   * @param {string} color
   * @param {string} size
   * @param {number} bucketSize
   * @return {!Array<!Array<string|number>>} A 2d array representing the data
   *     that will be visualized in the claibration plot.
   * @private
   */
  computePlotData_: function(
      buckets, header, fit, scale, color, size, bucketSize) {
    const plotData = [header];
    tfma.BucketsWrapper.getCalibrationPlotData(
        buckets, fit, scale, color, size, bucketSize, plotData);
    return plotData;
  }
});
