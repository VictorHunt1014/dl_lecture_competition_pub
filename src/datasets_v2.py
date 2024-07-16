import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler


class ThingsMEGDataset(Dataset):
    def __init__(
            self, 
            split: str, data_dir: str = "data", 
            resample_rate: int = None, 
            is_normalize: bool = False,
            augment: bool = True
        ) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.resample_rate = resample_rate
        self.is_normalize = is_normalize
        self.augment = augment
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        if self.resample_rate is not None and self.split == "train":
            self.X = self.resample_data(self.X)
        
        if self.is_normalize and self.split == "train":
            self.X = self.normalize_data(self.X)
            # self.X = self.normalize_data_per_subsject(self.X, self.subject_idxs)
        
        if self.augment and self.split == "train":
            self.X = self.augment_data(self.X)

    def resample_data(self, data):
        original_sample_rate = data.shape[2]
        num_samples = int((self.resample_rate / original_sample_rate) * data.shape[2])
        resampled_data = resample(data.numpy(), num=num_samples, axis=2)
        return torch.tensor(resampled_data, dtype=torch.float32)

    def normalize_data(self, data):
        scaler = StandardScaler()
        # データの形状を (samples * channels, time) に変換
        data = data.view(-1, data.size(-1)).numpy()  # (samples * channels, time)
        data = scaler.fit_transform(data)
        # データの形状を (samples, channels, time) に戻す
        num_samples, num_channels, num_timepoints = len(self.X), self.X.shape[1], data.shape[1]
        data = data.reshape(num_samples, num_channels, num_timepoints)
        return torch.tensor(data, dtype=torch.float32)

    def normalize_data_per_subject(self, data, subject_idxs):
        unique_subjects = torch.unique(subject_idxs)
        normalized_data = data.clone()
        
        for subject in unique_subjects:
            subject_data = data[subject_idxs == subject]
            scaler = StandardScaler()
            subject_data = subject_data.view(-1, subject_data.size(-1)).numpy()
            subject_data = scaler.fit_transform(subject_data)
            subject_data = subject_data.reshape(-1, data.shape[1], data.shape[2])
            normalized_data[subject_idxs == subject] = torch.tensor(subject_data, dtype=torch.float32)
        
        return normalized_data

    def augment_data(self, data):
        # データ拡張手法を実装（例：ランダムノイズ追加、時間シフト）
        if np.random.rand() > 0.5:
            data = data + 0.01 * torch.randn_like(data)
        return data

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
    
    @property
    def num_subjects(self) -> int:
        return len(torch.unique(self.subject_idxs))