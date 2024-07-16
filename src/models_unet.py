import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class UNetConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.encoder1 = ConvBlock(in_channels, hid_dim)
        self.encoder2 = ConvBlock(hid_dim, hid_dim * 2)
        self.encoder3 = ConvBlock(hid_dim * 2, hid_dim * 4)

        self.middle = ConvBlock(hid_dim * 4, hid_dim * 8)

        self.decoder3 = ConvBlock(hid_dim * 8 + hid_dim * 4, hid_dim * 4)
        self.decoder2 = ConvBlock(hid_dim * 4 + hid_dim * 2, hid_dim * 2)
        self.decoder1 = ConvBlock(hid_dim * 2 + hid_dim, hid_dim)

        self.head = nn.Sequential(
            nn.Conv1d(hid_dim, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        enc1 = self.encoder1(X)
        enc2 = self.encoder2(F.max_pool1d(enc1, 2))
        enc3 = self.encoder3(F.max_pool1d(enc2, 2))

        mid = self.middle(F.max_pool1d(enc3, 2))

        dec3 = self.decoder3(torch.cat([self._match_tensor_size(F.interpolate(mid, scale_factor=2, mode='nearest'), enc3), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self._match_tensor_size(F.interpolate(dec3, scale_factor=2, mode='nearest'), enc2), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([self._match_tensor_size(F.interpolate(dec2, scale_factor=2, mode='nearest'), enc1), enc1], dim=1))

        return self.head(dec1)

    def _match_tensor_size(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """Adjust tensor1 to match the size of tensor2"""
        if tensor1.size(2) > tensor2.size(2):
            return tensor1[:, :, :tensor2.size(2)]
        elif tensor1.size(2) < tensor2.size(2):
            padding = tensor2.size(2) - tensor1.size(2)
            return F.pad(tensor1, (0, padding))
        else:
            return tensor1

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.3,  # ドロップアウト率
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)