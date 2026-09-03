import torch

from data import Test_DS, train_DS


def get_dataset(batch_size):
    train_DL = torch.utils.data.DataLoader(train_DS, shuffle=True, batch_size=batch_size,
                                           num_workers=4, pin_memory=True, persistent_workers=True)
    Test_DL = torch.utils.data.DataLoader(Test_DS, shuffle=True, batch_size=batch_size,
                                          num_workers=2, pin_memory=True, persistent_workers=True)
    return train_DL, Test_DL
