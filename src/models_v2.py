import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(self, num_classes, seq_len, in_channels, num_subjects, hid_dim=128):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, hid_dim, num_subjects)
        self.conv2 = ConvBlock(hid_dim, hid_dim, num_subjects)
        self.conv3 = ConvBlock(hid_dim, hid_dim, num_subjects)
        self.subject_embedding = nn.Embedding(num_subjects, hid_dim)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim * 2, num_classes),
        )

    def forward(self, X, subject_idxs):
        batch_size, _, seq_len = X.size()
        X = self.conv1(X, subject_idxs)
        X = self.conv2(X, subject_idxs)
        X = self.conv3(X, subject_idxs)
        subject_emb = self.subject_embedding(subject_idxs).unsqueeze(-1)
        subject_emb = subject_emb.expand(batch_size, -1, seq_len)  # seq_lenに合わせて繰り返す
        X = torch.cat([X, subject_emb], dim=1)
        return self.head(X)
    

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_subjects, kernel_size=3, p_drop=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = SubjectBatchNorm1d(num_features=out_dim, num_subjects=num_subjects)
        self.batchnorm1 = SubjectBatchNorm1d(num_features=out_dim, num_subjects=num_subjects)
        self.batchnorm2 = SubjectBatchNorm1d(num_features=out_dim, num_subjects=num_subjects)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X, subject_idxs):
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X
        else:
            X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X, subject_idxs))
        X = self.conv1(X) + X
        X = F.gelu(self.batchnorm1(X, subject_idxs))
        X = self.conv2(X) + X
        X = F.gelu(self.batchnorm2(X, subject_idxs))
        return self.dropout(X)


class SubjectBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_subjects):
        super().__init__()
        self.num_subjects = num_subjects
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features) for _ in range(num_subjects)])

    def forward(self, x, subject_idxs):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)  # バッチ全体を float32 に変換
        
        out = torch.zeros_like(x, dtype=torch.float32)  # float32に変換
        for i in range(self.num_subjects):
            mask = (subject_idxs == i)
            if mask.sum() > 0:
                out[mask] = self.bns[i](x[mask])  # float32に変換
        return out