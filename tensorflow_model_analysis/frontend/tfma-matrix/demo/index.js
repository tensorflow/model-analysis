(() => {
  const simple = document.getElementById('simple');
  simple.data = {
    'R1': {
      'C1': {
        'value': 50,
        'tooltip': 'Click',
        'details': 'R1, C1',
      },
      'C2': {
        'value': 15,
        'tooltip': 'Click',
        'details': 'R1, C2',
      },
      'C3': {
        'value': 80,
        'tooltip': 'Click',
        'details': 'R1, C3',
      },
    },
    'R2': {
      'C1': {
        'value': 100,
        'tooltip': 'Click',
        'details': 'R2, C1',
      },
      'C3': {
        'value': 0,
        'tooltip': 'Click',
        'details': 'R2, C3',
      },
    },
  };

  const pivot = document.getElementById('pivot');
  pivot.data = {
    'R1': {
      'C1': {
        'value': 0.5,
        'tooltip': 'Click',
        'details': 'R1, C1',
      },
      'C2': {
        'value': 0.7,
        'tooltip': 'Click',
        'details': 'R1, C2',
      },
      'C3': {
        'value': -0.5,
        'tooltip': 'Click',
        'details': 'R1, C3',
      },
    },
    'R2': {
      'C1': {
        'value': -0.12,
        'tooltip': 'Click',
        'details': 'R2, C1',
      },
      'C2': {
        'value': 0.3,
        'tooltip': 'Click',
        'details': 'R2, C2',
      },
      'C3': {
        'value': 1,
        'tooltip': 'Click',
        'details': 'R2, C3',
      },
    },
    'R3': {
      'C1': {
        'value': -0.33,
        'tooltip': 'Click',
        'details': 'R2, C1',
      },
      'C2': {
        'value': -1,
        'tooltip': 'Click',
        'details': 'R2, C2',
      },
      'C3': {
        'value': 0,
        'tooltip': 'Click',
        'details': 'R2, C3',
      },
    },
  };

  const makeData = (row, column) => {
    const data = {};
    for (let i = 1; i <= row; i++) {
      const currentRow = {};
      for (let j = 1; j <= column; j++) {
        currentRow['C' + j] = {
          'value': Math.round(Math.random() * 100),
          'tooltip': 'Click',
          'details': 'R' + i + ', C' + j,
        };
      }
      data['R' + i] = currentRow;
    }
    return data;
  };

  const expandMe = document.getElementById('expand-me');
  expandMe.data = makeData(10, 8);

  const unexpandable = document.getElementById('unexpandable');
  unexpandable.data = makeData(10, 8);
  unexpandable.addEventListener('expand', (e) => {
    e.preventDefault();
  });

  const scale = document.getElementById('scale');
  scale.data = simple.data;
})();
