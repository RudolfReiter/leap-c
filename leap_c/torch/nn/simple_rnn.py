import torch
import torch.nn as nn

class SimpleRNNEncoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )

    def forward(self, x, h=None):
        """
        x:  Tensor of shape (B, 4)
        h:  hidden state of shape (1, B, 16) or None
        """
        B = x.size(0)

        # RNN expects sequence dimension, so add seq_len=1
        x = x.unsqueeze(1)  # (B, 1, 4)

        if h is None:
            h = torch.zeros(1, B, self.hidden_size, device=x.device)

        output, next_hidden = self.rnn(x, h)
        return next_hidden