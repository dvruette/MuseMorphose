from utils import pickle_load

import os, pickle, glob
import numpy as np
from collections import Counter

from constants import DEFAULT_DURATION_BINS

data_dir = os.getenv('REMI_DIR', 'lmd_remi')
file_pattern = os.getenv('FILE_PATTERN', '*.pkl')
polyph_out_dir = os.path.join(data_dir, 'attr_cls', 'polyph')
rhythm_out_dir = os.path.join(data_dir, 'attr_cls', 'rhythm')

rhym_intensity_bounds = [0.2, 0.25, 0.32, 0.38, 0.44, 0.5, 0.63]
polyphonicity_bounds = [2.63, 3.06, 3.50, 4.00, 4.63, 5.44, 6.44]

def compute_polyphonicity(events, n_bars):
  poly_record = np.zeros( (n_bars * 48,) )

  cur_bar, cur_pos = -1, -1
  for ev in events:
    if ev['name'] == 'Bar':
      cur_bar += 1
    elif ev['name'] == 'Beat':
      cur_pos = int(ev['value'])
    elif ev['name'] == 'Note_Duration':
      duration_cls = int(ev['value'])
      duration = DEFAULT_DURATION_BINS[duration_cls]
      # print(duration_cls, DEFAULT_DURATION_BINS[duration_cls])
      st = cur_bar * 48 + cur_pos
      poly_record[st:st + duration] += 1
  
  return poly_record

def get_onsets_timing(events, n_bars):
  onset_record = np.zeros( (n_bars * 48,) )

  cur_bar, cur_pos = -1, -1
  for ev in events:
    if ev['name'] == 'Bar':
      cur_bar += 1
    elif ev['name'] == 'Beat':
      cur_pos = int(ev['value'])
    elif ev['name'] == 'Note_Pitch':
      rec_idx = cur_bar * 48 + cur_pos
      onset_record[ rec_idx ] = 1

  return onset_record

if __name__ == "__main__":
  pieces = sorted([os.path.basename(f) for f in glob.glob(os.path.join(data_dir, file_pattern))])
  all_r_cls = []
  all_p_cls = []

  if not os.path.exists(polyph_out_dir):
    os.makedirs(polyph_out_dir)
  if not os.path.exists(rhythm_out_dir):
    os.makedirs(rhythm_out_dir)  

  for i, p in enumerate(pieces):
    print(f'\r{i+1:4d}/{len(pieces):4d}', end='')
    bar_pos, events = pickle_load(os.path.join(data_dir, p))
    events = events[ :bar_pos[-1] ]

    polyph_raw = np.reshape(
      compute_polyphonicity(events, n_bars=len(bar_pos)), (-1, 16)
    )
    rhythm_raw = np.reshape(
      get_onsets_timing(events, n_bars=len(bar_pos)), (-1, 16)
    )

    polyph_cls = np.searchsorted(polyphonicity_bounds, np.mean(polyph_raw, axis=-1)).tolist()
    rfreq_cls = np.searchsorted(rhym_intensity_bounds, np.mean(rhythm_raw, axis=-1)).tolist()

    pickle.dump(polyph_cls, open(os.path.join(
      polyph_out_dir, p), 'wb'
    ))
    pickle.dump(rfreq_cls, open(os.path.join(
      rhythm_out_dir, p), 'wb'
    ))

    all_r_cls.extend(rfreq_cls)
    all_p_cls.extend(polyph_cls)
  print()
  print ('[rhythm classes]', Counter(all_r_cls))
  print ('[polyph classes]', Counter(all_p_cls))