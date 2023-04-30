import torch
import numpy as np
from torch.utils.data import Dataset


class LoadDataset_from_numpy(Dataset):
    def __init__(self, np_files):
        super(LoadDataset_from_numpy, self).__init__()

        eeg = np.load(np_files[0])["eeg"]
        eog = np.load(np_files[0])["eog"]
        label = np.load(np_files[0])["y"]

        for np_file in np_files[1:]:
            eeg = np.vstack((eeg, np.load(np_file)["eeg"]))
            eog = np.vstack((eog, np.load(np_file)["eog"]))
            label = np.append(label, np.load(np_file)["y"])

        self.len = eeg.shape[0]
        self.eeg = torch.from_numpy(eeg).permute(0, 2, 1)
        self.eog = torch.from_numpy(eog).permute(0, 2, 1)
        self.label = torch.from_numpy(label).long()

    def __getitem__(self, index):
        return self.eeg[index], self.eog[index], self.label[index]

    def __len__(self):
        return self.len


def data_generator_np(training_files, test_files, batch_size):
    train_dataset = LoadDataset_from_numpy(training_files)
    test_dataset = LoadDataset_from_numpy(test_files)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, test_loader
