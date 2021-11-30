import glob
import os
import pickle
from input_representation import InputRepresentation

folder = os.getenv('LMD_DIR', './lmd_full/0')
out_folder = os.getenv('REMI_DIR', './lmd_remi')

if __name__ == '__main__':
  os.makedirs(out_folder, exist_ok=True)

  midi_files = glob.glob(os.path.join(folder, '**', '*.mid'), recursive=True)

  idx = 0
  for i, file in enumerate(midi_files):
    print(f'\r{i+1:4d}/{len(midi_files):4d}', end='')
    try:
      ir = InputRepresentation(file)
    except Exception:
      continue

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

