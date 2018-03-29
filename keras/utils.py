import os
import re

weight_file_re = re.compile(r'.*(\d+)\.h5$')

def load_weights(model, prefix):
  weight_files = [
    name for name in os.listdir('./out') if name.endswith('.h5') and
      name.startswith(prefix)
  ]

  saved_epochs = []
  for name in weight_files:
    match = weight_file_re.match(name)
    if match == None:
      continue
    saved_epochs.append({ 'name': name, 'epoch': int(match.group(1)) })
  saved_epochs.sort(key=lambda entry: entry['epoch'], reverse=True)

  for save in saved_epochs:
    try:
      model.load_weights(os.path.join('./out', save['name']))
    except IOError:
      continue
    start_epoch = save['epoch']
    print("Loaded weights from " + save['name'])
    break

