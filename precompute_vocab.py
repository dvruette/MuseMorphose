import vocab
import pickle

out_file = './pickles/remi_vocab_lmd.pkl'

voc = vocab.RemiVocab()

events = voc.vocab.get_itos()
events.extend(['Bar_None', 'EOS_None'])

idx2event = { i: event for i, event in enumerate(events) }
event2idx = { event: i for i, event in enumerate(events) }

pickle.dump((event2idx, idx2event), open(out_file, 'wb'))