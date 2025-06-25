import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os


class RML2018Dataset(Dataset):
    def __init__(self, data_path, split='train', transform=None, normalize=True):
        """
        RML2018数据集加载器

        Args:
            data_path: 数据文件路径 (pkl文件)
            split: 'train', 'val', 'test'
            transform: 数据变换
            normalize: 是否归一化到[-1, 1]
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.normalize = normalize

        # 加载数据
        self.load_data()

    def load_data(self):
        """加载RML2018数据"""
        # 假设数据格式为: dict with keys 'data' and 'labels'
        # data shape: (N, 2, 1024) - N个样本，每个样本2通道(I/Q)，长度1024
        # labels: (N,) - 对应的调制类型标签

        with open(self.data_path, 'rb') as f:
            data_dict = pickle.load(f)

        data = np.array(data_dict['data'])  # shape: (N, 2, 1024)
        labels = np.array(data_dict['labels'])  # shape: (N,)

        # 数据划分
        total_samples = len(data)
        if self.split == 'train':
            start_idx = 0
            end_idx = int(0.8 * total_samples)
        elif self.split == 'val':
            start_idx = int(0.8 * total_samples)
            end_idx = int(0.9 * total_samples)
        else:  # test
            start_idx = int(0.9 * total_samples)
            end_idx = total_samples

        self.data = data[start_idx:end_idx]
        self.labels = labels[start_idx:end_idx]

        # 归一化到[-1, 1]
        if self.normalize:
            # 假设原始数据范围合理，可以直接除以最大绝对值
            max_val = np.max(np.abs(self.data))
            self.data = self.data / max_val
            # 或者使用标准化
            # self.data = (self.data - np.mean(self.data)) / np.std(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].astype(np.float32)
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.long)


def get_dataloader(data_path, batch_size=64, num_workers=4, split='train'):
    """获取数据加载器"""
    dataset = RML2018Dataset(data_path, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
