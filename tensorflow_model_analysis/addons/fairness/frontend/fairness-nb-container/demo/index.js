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
(() => {
  const createSliceMetrics1 = (slice_name) => {
    if (slice_name.includes('omitted')) {
      // Example count for this slice key is lower than the minimum required
      // value: 10. No data is aggregated
      return {
        '__ERROR__': {
          'bytesValue':
              'RXhhbXBsZSBjb3VudCBmb3IgdGhpcyBzbGljZSBrZXkgaXMgbG93ZXIgdGhhbiB0aGUgbWluaW11\nbSByZXF1aXJlZCB2YWx1ZTogMTAuIE5vIGRhdGEgaXMgYWdncmVnYXRlZA=='
        }
      };
    }
    return {
      'accuracy': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      'post_export_metrics/positive_rate@0.50': {
        'doubleValue': NaN,
      },
      'post_export_metrics/negative_rate@0.50': {
        'doubleValue': Math.random(),
      },
      'post_export_metrics/positive_rate@0.60': {
        'boundedValue': {
          'lowerBound': NaN,
          'upperBound': NaN,
          'value': NaN,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      'post_export_metrics/negative_rate@0.60': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      'post_export_metrics/positive_rate@0.70': {
        'doubleValue': NaN,
      },
      'post_export_metrics/negative_rate@0.70': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      'post_export_metrics/positive_rate@0.80': {
        'boundedValue': {
          'lowerBound': NaN,
          'upperBound': NaN,
          'value': NaN,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      'post_export_metrics/negative_rate@0.80': {
        'doubleValue': Math.random(),
      },
      'post_export_metrics/example_count': {
        'boundedValue': {
          'lowerBound': Math.random() * 10 + 90,
          'upperBound': Math.random() * 10 + 110,
          'value': Math.random() * 10 + 100,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      'totalWeightedExamples': {'doubleValue': 2000 * (Math.random() + 0.8)},

      // These two metrics only exist inside this eval result
      'a_metric_only_in_eval1': {'doubleValue': 2000 * (Math.random() + 0.8)},
      'another_metric_only_in_eval1': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      // CI not computed because only 86 samples were non-empty. Expected 100.
      '__ERROR__': {
        'bytesValue':
            'Q0kgbm90IGNvbXB1dGVkIGJlY2F1c2Ugb25seSA4NiBzYW1wbGVzIHdlcmUgbm9uLWVtcHR5LiBFeHBlY3RlZCAxMDAu'
      }
    };
  };

  const createSliceMetrics2 = () => {
    return {
      'accuracy': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      // doubleValue NaN vs doubleValue NaN
      'post_export_metrics/positive_rate@0.50': {
        'doubleValue': NaN,
      },
      // doubleValue vs doubleValue
      'post_export_metrics/negative_rate@0.50': {
        'doubleValue': Math.random(),
      },
      // boundedValue NaN vs boundedValue NaN
      'post_export_metrics/positive_rate@0.60': {
        'boundedValue': {
          'lowerBound': NaN,
          'upperBound': NaN,
          'value': NaN,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      // boundedValue vs boundedValue
      'post_export_metrics/negative_rate@0.60': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      // Double NaN vs boundedValue
      'post_export_metrics/positive_rate@0.70': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      // boundedValue vs doubleValue
      'post_export_metrics/negative_rate@0.70': {
        'doubleValue': Math.random(),
      },
      // doubleValue NaN vs Double
      'post_export_metrics/positive_rate@0.80': {
        'doubleValue': Math.random(),

      },
      // doubleValue vs boundedValue
      'post_export_metrics/negative_rate@0.80': {
        'boundedValue': {
          'lowerBound': Math.random() * 0.3,
          'upperBound': Math.random() * 0.3 + 0.6,
          'value': Math.random() * 0.3 + 0.3,
          'methodology': 'POISSON_BOOTSTRAP'
        }
      },
      'post_export_metrics/example_count':
          {'doubleValue': Math.floor(Math.random() * 100)},
      'totalWeightedExamples': {'doubleValue': 2000 * (Math.random() + 0.8)}
    };
  };

  const SLICES1 = [
    'Overall',
    'Slice:unique',
    'Sex:Male',
    'Sex:Female',
    'sexual_orientation:bisexual',
    'sexual_orientation:heterosexual',
    'sexual_orientation:homosexual',
    'Sex:Transgender',
    'race:asian',
    'race:latino',
    'race:black',
    'race:white',
    'religion:atheist',
    'religion:buddhist',
    'religion:christian',
    'religion:hindu',
    'religion:jewish',
    'religion:muslim',
    'religion:omitted',
  ];
  const input1 = SLICES1.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': createSliceMetrics1(slice),
    };
  });

  const SLICES2 = [
    'Overall',
    'Sex:Male',
    'Sex:Female',
    'sexual_orientation:bisexual',
    'sexual_orientation:heterosexual',
    'sexual_orientation:homosexual',
    'Sex:Transgender',
    'race:asian',
    'race:latino',
    'race:black',
    'race:white',
    'religion:atheist',
    'religion:buddhist',
    'religion:christian',
    'religion:hindu',
    'religion:jewish',
    'religion:omitted',
  ];
  const input2 = SLICES2.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': createSliceMetrics2(),
    };
  });

  const input3 = SLICES1.map((slice) => {
    return {
      'slice': slice,
      'sliceValue': slice.split(':')[1] || 'Overall',
      'metrics': createSliceMetrics1(slice),
    };
  });

  const element = document.getElementsByTagName('fairness-nb-container')[0];
  element.slicingMetrics = input3;

  const element_compare =
      document.getElementsByTagName('fairness-nb-container')[1];
  element_compare.slicingMetrics = input1;
  element_compare.slicingMetricsCompare = input2;
  element_compare.evalName = 'Eval1';
  element_compare.evalNameCompare = 'Eval2';
  element_compare.availbleEvaluationRuns = ['1', '2', '3'];
})();
