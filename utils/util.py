import json
import os
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path
from collections import OrderedDict
from itertools import repeat


def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"utils/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)
    npzfiles = np.asarray(files, dtype='<U200')[r_permute]
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data


def load_folds_data(np_data_path, n_folds):
    r_p_path = r"utils/r_permute_20.npy"
    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
    else:
        print("============== ERROR =================")

    files_dict = dict()
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    for i in files:
        file_name = os.path.split(i)[-1]
        file_num = file_name[3:5]
        if file_num not in files_dict:
            files_dict[file_num] = [i]
        else:
            files_dict[file_num].append(i)

    files_pairs = []
    for key in files_dict:
        files_pairs.append(files_dict[key])
    files_pairs = np.array(files_pairs, dtype=tuple)
    files_pairs = files_pairs[r_permute]

    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        subject_files = [item for sublist in subject_files for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    '''wrapper function for endless data loader.'''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
