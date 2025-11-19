import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSetLayer(nn.Module):
    """
    Deep Set (Zaheer et al.) layer.

    Inputs:
        x:    (B, N, 4)    set elements
        mask: (B, N)       1/True for valid, 0/False for invalid elements

    Output:
        out:  (B, 16)      set embedding
    """
    def __init__(self,
                 elem_dim: int = 4,
                 phi_hidden: int = 16,
                 rho_hidden: int = 16,
                 out_dim: int = 16):
        super().__init__()

        # φ: element-wise embedding network
        self.phi = nn.Sequential(
            nn.Linear(elem_dim, phi_hidden),
            nn.ReLU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
        )

        # ρ: post-aggregation network
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden, rho_hidden),
            nn.ReLU(),
            nn.Linear(rho_hidden, out_dim),
        )

    def forward(self, x, mask):
        """
        x:    (B, N, 4)
        mask: (B, N)  bool or {0,1}
        """
        # Ensure mask is float and has a channel dimension for broadcasting
        mask = mask.float().unsqueeze(-1)        # (B, N, 1)

        # φ applied independently to each element
        phi_x = self.phi(x)                      # (B, N, H_phi)

        # Zero-out invalid elements before aggregation
        phi_x_masked = phi_x * mask              # (B, N, H_phi)

        # Sum aggregation over N (set elements)
        set_repr = phi_x_masked.sum(dim=1)       # (B, H_phi)

        # ρ maps to final embedding
        out = self.rho(set_repr)                 # (B, 16)
        return out


# Example usage
if __name__ == "__main__":
    B, N = 8, 10
    x = torch.randn(B, N, 4)
    # Suppose first k_i entries are valid for each batch; simple example:
    mask = torch.zeros(B, N)
    mask[:, :5] = 1.0  # only first 5 valid

    model = DeepSetLayer(out_dim=16)
    emb = model(x, mask)
    print(emb.shape)  # -> torch.Size([8, 16])
