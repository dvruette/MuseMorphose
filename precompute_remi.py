import glob
import os
import pickle
from input_representation import InputRepresentation

folder = './lmd_full/0'
out_folder = './lmd_remi'

os.makedirs(out_folder, exist_ok=True)

midi_files = glob.glob(os.path.join(folder, '**', '*.mid'), recursive=True)

try:
  idx = 0
  for i, file in enumerate(midi_files):
    print(f'\r{i+1:4d}/{len(midi_files):4d}', end='')
    ir = InputRepresentation(file)
    ts = ir.pm.time_signature_changes
    if len(ts) != 1:
      continue
    if ts[0].numerator != 4 or ts[0].denominator != 4:
      continue
    try:
      events = ir.get_remi_events()
    except:
      continue

    bars = [i for i, event in enumerate(events) if event['name'] == 'Bar']
    if len(bars) == 0:
      continue

    idx += 1
    out_file = os.path.join(out_folder, f"{idx}.pkl")
    pickle.dump((bars, events), open(out_file, 'wb'))

  print()
finally:
  if idx > 0:
    n_train = int(0.8 * idx)
    n_val = int(0.1 * idx)

    train_files = [f"{i+1}.pkl" for i in range(n_train)]
    val_files = [f"{i+1}.pkl" for i in range(n_train, n_train + n_val)]
    test_files = [f"{i+1}.pkl" for i in range(n_train + n_val, idx)]

    pickle.dump(train_files, open('./pickles/train_pieces_lmd.pkl', 'wb'))
    pickle.dump(val_files, open('./pickles/val_pieces_lmd.pkl', 'wb'))
    pickle.dump(test_files, open('./pickles/test_pieces_lmd.pkl', 'wb'))

