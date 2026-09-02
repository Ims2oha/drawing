"""DataLoaders for the QuickDraw datasets built in `data.py`."""

import torch

from data import Test_DS, train_DS


def get_dataset(batch_size):
    train_DL = torch.utils.data.DataLoader(train_DS, shuffle=True, batch_size=batch_size)
    Test_DL = torch.utils.data.DataLoader(Test_DS, shuffle=True, batch_size=batch_size)
    return train_DL, Test_DL
