import sys, os, random, time
from copy import deepcopy

from torch.nn.functional import pad
sys.path.append('./model')

from dataloader import REMIFullSongTransformerDataset
from model.musemorphose import MuseMorphose

import torch
import yaml
import numpy as np
import glob

config_path = sys.argv[1]
config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

device = config['training']['device']
vocab_path = config['data']['vocab_path']
data_dir = config['data']['data_dir'].replace('$SCRATCH', os.getenv('SCRATCH', '.'))
pieces = glob.glob(os.path.join(data_dir, '*.pkl'))

n_pieces = len(pieces)
n_val_pieces = max(1024, int(0.1 * n_pieces))
val_pieces = pieces[-n_val_pieces:]

ckpt_path = sys.argv[2]
out_file = sys.argv[3]
n_pieces = int(sys.argv[4])


def ensure_out_file(file=out_file):
  # create file if it doesn't exist
  if not os.path.exists(file):
    with open(file, 'w') as f:
      f.write("id,ppl,total_entopy,n_tokens\n")


def validate(model, batch, context_size):
  model.eval()

  start_time = time.time()

  print (f"[info] validating {batch['piece_id']} ...")
  with torch.no_grad():
    batch_enc_inp = torch.tensor(batch['enc_input'], device=device).t().unsqueeze(1)
    batch_dec_inp = torch.tensor(batch['dec_input'], device=device).unsqueeze(1)
    batch_dec_tgt = torch.tensor(batch['dec_target'], device=device).unsqueeze(1)
    batch_inp_bar_pos = torch.tensor(batch['bar_pos'], device=device).unsqueeze(0)
    batch_padding_mask = torch.tensor(batch['enc_padding_mask'], device=device).unsqueeze(0)
    batch_rfreq_cls = torch.tensor(batch['rhymfreq_cls'], device=device).unsqueeze(1)
    batch_polyph_cls = torch.tensor(batch['polyph_cls'], device=device).unsqueeze(1)

    tot_ent = 0
    tot_len = 0

    ignore_idx = model.n_token - 1

    for i in range(min(batch_dec_inp.size(0), context_size), batch_dec_inp.size(0) + 1):
      # print(f"\r[info] step {i:4d}/{batch_dec_inp.size(0)}", end='')
      start, end = max(0, i - context_size), i
      dec_inp = batch_dec_inp[start:end]
      rfreq_cls = batch_rfreq_cls[start:end]
      polyph_cls = batch_polyph_cls[start:end]

      _, _, logits = model(
        batch_enc_inp, dec_inp, 
        batch_inp_bar_pos, rfreq_cls, polyph_cls,
        padding_mask=batch_padding_mask
      )

      y = batch_dec_tgt[start:end]

      log_pr = logits.log_softmax(dim=-1)
      log_pr[y == ignore_idx] = 0 # log(pr) = log(1) for padding
      log_pr = torch.gather(log_pr, -1, y.unsqueeze(-1)).squeeze(-1)

      if start == 0:
        tot_ent += -log_pr.sum()
        tot_len += (y != ignore_idx).sum()
      else:
        tot_ent += -log_pr[-1].item()
        tot_len += 1
    
    if tot_len > 0:
      ppl = (tot_ent / tot_len).exp()

      ensure_out_file()
      with open(out_file, 'a') as f:
        f.write(f"{batch['piece_id']},{ppl},{tot_ent},{tot_len}\n")
    else:
      ppl = np.nan
    
    tot_time = time.time() - start_time
    per_token = tot_time / tot_len if tot_len > 0 else np.nan
    # print('\r', end='')
    print(f"[info] PPL: {ppl:.3f}, took {tot_time:.1f}s ({per_token:.2f}s per token, {tot_len} tokens)")


if __name__ == "__main__":
  dset = REMIFullSongTransformerDataset(
    data_dir, vocab_path, 
    do_augment=False,
    model_enc_seqlen=config['data']['enc_seqlen'], 
    model_dec_seqlen=config['generate']['dec_seqlen'],
    model_max_bars=config['generate']['max_bars'],
    pieces=val_pieces,
    pad_to_same=False
  )
  
  mconf = config['model']
  model = MuseMorphose(
    mconf['enc_n_layer'], mconf['enc_n_head'], mconf['enc_d_model'], mconf['enc_d_ff'],
    mconf['dec_n_layer'], mconf['dec_n_head'], mconf['dec_d_model'], mconf['dec_d_ff'],
    mconf['d_latent'], mconf['d_embed'], dset.vocab_size,
    d_polyph_emb=mconf['d_polyph_emb'], d_rfreq_emb=mconf['d_rfreq_emb'],
    cond_mode=mconf['cond_mode']
  ).to(device)
  model.eval()
  model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

  pieces = random.sample(range(len(dset)), n_pieces)
  print ('[sampled pieces]', pieces)

  for p in pieces:
    validate(model, dset[p], context_size=config['generate']['max_input_dec_seqlen'])