import struct

def load_list(f):
  list_len = struct.unpack('<i', f.read(4))[0]
  out = []
  for i in range(list_len):
    out.append(struct.unpack('f', f.read(4))[0])
  return out

def load(fname):
  with open(fname, 'rb') as f:
    step = struct.unpack('<i', f.read(4))[0]

    train = {
      'positives': load_list(f),
      'negatives': load_list(f),
    }
    validate = {
      'positives': load_list(f),
      'negatives': load_list(f),
    }

    return { 'step': step, 'train': train, 'validate': validate }
