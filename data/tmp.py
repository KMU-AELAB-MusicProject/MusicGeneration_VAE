import os
from config import *
import numpy as np
import pickle
import scipy.sparse
import time


for i in range(11):
    loaded = np.load(os.path.join(DATA_PATH, 'bar_data', 'bar_data{}.npz'.format(i)))

for i in range(11):
    loaded = np.load(os.path.join(DATA_PATH, 'phrase_data', 'phrase_data{}.npz'.format(i)))

os._exit(1)


tmp = []
cnt = 1
print(len(os.listdir(NP_FILE_PATH)))
for file in os.listdir(NP_FILE_PATH):
    print(cnt, file, len(tmp))
    with open(os.path.join(NP_FILE_PATH, file), 'rb') as f:
        np_data = pickle.load(f)
        note_data, number_data = np_data
        tmp.extend(note_data)
    cnt += 1

print(len(tmp))