import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Chop off padding at the end to maintain causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection: match channel dims if needed
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else
            None
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TrajectoryTCN(nn.Module):
    """
    Temporal Convolution Network for trajectories.

    Input:  (B, K, 3)  -> positions over time
    Output: (B, embed_dim)
    """
    def __init__(
        self,
        embed_dim: int,
        num_channels=(16,16),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_channels = 3  # xyz
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], embed_dim)

    def forward(self, x):
        """
        x: (B, K, 3)
        """
        # (B, K, 3) -> (B, C_in=3, L=K)
        x = x.transpose(1, 2)

        # TCN: (B, C_out, L)
        y = self.tcn(x)

        # Global average pooling over time L
        y = y.mean(dim=-1)  # (B, C_out)

        # Final embedding
        emb = self.fc(y)    # (B, embed_dim)
        return emb


import torch

def flatten_and_concat(*tensors):
    """
    Flattens each input tensor except the batch dimension and concatenates them.

    Args:
        *tensors: any number of tensors shaped (B, ...)

    Returns:
        Tensor of shape (B, M) where M is the sum of flattened sizes.
    """
    flat_list = []
    for t in tensors:
        B = t.shape[0]
        flat = t.reshape(B, -1)   # flatten all but batch
        flat_list.append(flat)

    return torch.cat(flat_list, dim=-1)  # concatenate on feature dim

# Example usage:
if __name__ == "__main__":
    B, K = 8, 50
    traj = torch.randn(B, K, 3)  # batch of trajectories

    model = TrajectoryTCN(embed_dim=16)
    out = model(traj)
    print(out.shape)  # torch.Size([8, 16])
