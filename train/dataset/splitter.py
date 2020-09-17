import pandas as pd
from tqdm import trange
from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]

batch_size = 2000

for _,filename in zip(trange(len(onlyfiles)), onlyfiles):
  if filename == 'splitter.py': continue
  for i in range(5):
    df = pd.read_csv(filename, header=None, nrows=batch_size, skiprows=int(i*batch_size), usecols=list(range(17))[1:])
    if i == 0:
      df.to_csv('split/val_'+filename, mode='a', header=False)
    else:
      df.to_csv('split/train_'+filename, mode='a', header=False)
    del df
